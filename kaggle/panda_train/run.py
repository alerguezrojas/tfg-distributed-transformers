"""Kaggle (2x T4): PANDA trainings with DDP and test evaluation, one after the other.

Reads the mosaics, ``splits.csv`` and ``train.csv`` written by the ``paravit-panda-mosaicos``
kernel (attached as a kernel source), trains each experiment of ``EXPERIMENTS`` with
``scripts/train_ddp.py`` (2 GPUs, ``batch_size`` per GPU) and evaluates its best checkpoint on
test with ``scripts/eval.py --save-predictions``. Everything useful ends in /kaggle/working:

    results/<name>/test_preds.npz, best checkpoint, eval log, train config
    logs/            the run logs (same layout as the local logs/ folder)
    timings.csv      wall time per experiment, to check the performance predictor

Each experiment is ``(name, base config in configs/, overrides)``. The global batch stays at 16
(8 per GPU) as in the local runs, so only the resolution changes between ``*_512`` and ``*_1024``.
Pushed with ``scripts/kaggle_push.sh kaggle/panda_train`` (see ``kaggle/README.md``).
"""
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/alerguezrojas/tfg-distributed-transformers.git"
REF = os.environ.get("PARAVIT_REF", "develop")
REPO = Path("/tmp/repo")
ROOT = Path("/tmp/panda")
OUT = Path("/kaggle/working")
MOSAICS = "mosaics_l1_16x256"
SPLITS_MD5 = "76dcfaff305092dbe98b8387df19a894"   # splits.csv of the local runs (same partition)


def res(size: int) -> dict:
    return {"data": {"image_size": size}, "model": {"img_size": size}}


EXPERIMENTS = [
    ("panda_ddp_512", "configs/train_panda.yaml", res(512)),           # control: Kaggle = local?
    ("panda_ddp_1024", "configs/train_panda.yaml", res(1024)),
    ("panda_ordinal_ddp_1024", "configs/train_panda_ordinal.yaml", res(1024)),
]
COMMON = {
    "data": {"num_workers": 4},
    "training": {"batch_size": 8},                                     # per GPU -> global 16
    "output": {"env": "kaggle"},
}


def sh(*cmd: str, log: Path | None = None, env: dict | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    if log is None:
        return subprocess.run(cmd, check=False, env=env).returncode
    with open(log, "w") as f:
        proc = subprocess.run(cmd, check=False, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(log.read_text()[-1500:], flush=True)
    return proc.returncode


def find_mosaics() -> Path:
    for splits in sorted(Path("/kaggle/input").rglob("splits.csv")):
        if (splits.parent / MOSAICS).is_dir():
            return splits.parent
    raise SystemExit("mosaics not found: attach the paravit-panda-mosaicos kernel output")


def deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_config(repo: Path, name: str, base: str, overrides: dict) -> dict:
    """The YAML of one experiment: the repo config + COMMON + its overrides (pure, tested)."""
    import yaml
    cfg = yaml.safe_load((repo / base).read_text())
    deep_update(cfg, COMMON)
    deep_update(cfg, overrides)
    cfg["data"]["root"] = str(ROOT)
    cfg.setdefault("checkpoint", {})["dir"] = f"/tmp/ckpt/{name}"
    return cfg


def write_config(name: str, base: str, overrides: dict) -> Path:
    import yaml
    path = REPO / "configs" / f"_kaggle_{name}.yaml"
    path.write_text(yaml.safe_dump(build_config(REPO, name, base, overrides), sort_keys=False))
    return path


def best_checkpoint(name: str) -> Path | None:
    found = sorted(Path(f"/tmp/ckpt/{name}").rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    best = [p for p in found if p.name == "best.pt"]
    return (best or found or [None])[-1]


def main() -> None:
    src = find_mosaics()
    digest = hashlib.md5((src / "splits.csv").read_bytes()).hexdigest()
    n_mosaics = sum(1 for _ in (src / MOSAICS).glob("*.jpg"))
    print(f"mosaics: {src} ({n_mosaics} jpg) | splits.csv md5 {digest}", flush=True)
    if digest != SPLITS_MD5:
        raise SystemExit("splits.csv differs from the local one: the runs would not be comparable")
    sh("git", "clone", "--depth", "1", "--branch", REF, REPO_URL, str(REPO))
    sh(sys.executable, "-m", "pip", "install", "-q", "timm", "nvidia-ml-py", "torchinfo")
    sh("nvidia-smi")
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "splits.csv", MOSAICS):
        (ROOT / name).symlink_to(src / name)
    os.chdir(REPO)
    timings = []
    for name, base, overrides in EXPERIMENTS:
        cfg = write_config(name, base, overrides)
        dest = OUT / "results" / name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg, dest / "config.yaml")
        t0 = time.time()
        rc = sh(sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
                "scripts/train_ddp.py", "--config", str(cfg),
                "--precision", "amp", "--monitor", log=dest / "train.log")
        train_s = time.time() - t0
        ckpt = best_checkpoint(name)
        rc_eval = -1
        if rc == 0 and ckpt is not None:
            shutil.copy(ckpt, dest / ckpt.name)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            rc_eval = sh(sys.executable, "scripts/eval.py", "--checkpoint", str(ckpt),
                         "--config", str(cfg), "--split", "test",
                         "--save-predictions", str(dest / "test_preds.npz"),
                         log=dest / "eval.log", env=env)
        timings.append({"name": name, "train_rc": rc, "eval_rc": rc_eval,
                        "train_min": round(train_s / 60, 1), "checkpoint": ckpt.name if ckpt else ""})
        with open(OUT / "timings.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(timings[0]))
            writer.writeheader()
            writer.writerows(timings)
        if (REPO / "logs").is_dir():
            shutil.copytree(REPO / "logs", OUT / "logs", dirs_exist_ok=True)
    print(open(OUT / "timings.csv").read(), flush=True)


if __name__ == "__main__":
    main()
