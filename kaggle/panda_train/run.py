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


COMMON_BATCH_PER_GPU = 8                                               # x 2 GPUs = 16


def deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def res(size: int, seed: int = 42) -> dict:
    return {"data": {"image_size": size}, "model": {"img_size": size}, "training": {"seed": seed}}


def model(name: str, seed: int = 42) -> dict:
    """Overrides for another timm model at 512 px (the base config's resolution)."""
    return {"model": {"name": name}, "training": {"seed": seed}}


def single(overrides: dict) -> dict:
    """The same experiment on ONE T4: the whole global batch (16) on that GPU."""
    return deep_update(overrides, {"training": {"batch_size": 2 * COMMON_BATCH_PER_GPU}})


# One batch per kernel run (a session lasts at most 12 h; a 1024-px run takes ~2.9 h on 2x T4).
# The finished batches stay here as the record of what was run; BATCH picks the next one.
BATCHES = {
    # 20/11/2026: Kaggle = local at 512 (control) and the first look at 1024.
    "resolution": [
        ("panda_ddp_512", "configs/train_panda.yaml", res(512)),
        ("panda_ddp_1024", "configs/train_panda.yaml", res(1024)),
        ("panda_ordinal_ddp_1024", "configs/train_panda_ordinal.yaml", res(1024)),
    ],
    # 20/11/2026 (2nd): seeds 43 and 44 of the base at 1024 (seed 42 gave 0.821 vs 0.794 +- 0.012).
    # Relaunched after PR #216: the first launch ran with the DistributedSampler always on seed 0
    # (same data order for every seed) and was stopped. The seed-42 run of "resolution" also used
    # sampler seed 0, which is just one more data order.
    # Result (21/11/2026): test QWK 0.787 and 0.805 -> 0.804 +- 0.017 over 3 seeds, no significant
    # gain over 512 px (docs/datasets/panda.md).
    "seeds_1024": [
        ("panda_ddp_1024_s43", "configs/train_panda.yaml", res(1024, 43)),
        ("panda_ddp_1024_s44", "configs/train_panda.yaml", res(1024, 44)),
    ],
    # 22/11/2026: models that do not fit the local 8 GB GPU at 512 px with batch 16 (8 per T4
    # here), same recipe as the ViT-S base (LLRD normalised to 12 blocks, PR #234). One seed;
    # the local screening (efficientnet_b0, convnext_tiny, deit3_small, swin_tiny, resnet50,
    # efficientnetv2_s, eva02_small) runs at the same time on the local GPU.
    "models_512": [
        ("panda_ddp_vitb", "configs/train_panda.yaml",
         model("vit_base_patch16_224.augreg2_in21k_ft_in1k")),
        ("panda_ddp_convnext_s", "configs/train_panda.yaml", model("convnext_small.fb_in22k")),
        ("panda_ddp_swinv2_t", "configs/train_panda.yaml", model("swinv2_tiny_window8_256")),
        ("panda_ddp_effb3", "configs/train_panda.yaml", model("efficientnet_b3")),
        ("panda_ddp_maxvit_t", "configs/train_panda.yaml", model("maxvit_tiny_tf_512.in1k")),
    ],
    # Result (23/11/2026), test QWK with seed 42: SwinV2-T 0.835, ViT-B 0.823, ConvNeXt-S 0.811,
    # MaxViT-T 0.806, EfficientNet-B3 0.775 (base ViT-S 0.781, Swin-T 0.830); ~50 min per model.
    # 23/11/2026: seeds 43 and 44 of the two best (SwinV2-T, ViT-B), to confirm them and the
    # mixed-architecture ensemble with runs not used to pick it; and Swin-T with DDP on 2x T4,
    # the same run as the local single-GPU one (time and accuracy across strategies).
    "seeds_512": [
        ("panda_ddp_swinv2_t_s43", "configs/train_panda.yaml", model("swinv2_tiny_window8_256", 43)),
        ("panda_ddp_vitb_s43", "configs/train_panda.yaml",
         model("vit_base_patch16_224.augreg2_in21k_ft_in1k", 43)),
        ("panda_ddp_swinv2_t_s44", "configs/train_panda.yaml", model("swinv2_tiny_window8_256", 44)),
        ("panda_ddp_vitb_s44", "configs/train_panda.yaml",
         model("vit_base_patch16_224.augreg2_in21k_ft_in1k", 44)),
        ("panda_ddp_swin_t", "configs/train_panda.yaml", model("swin_tiny_patch4_window7_224")),
    ],
    # Result (23/11/2026), test QWK: SwinV2-T 0.824 / 0.802 and ViT-B 0.819 / 0.825 (s43 / s44);
    # Swin-T with DDP 0.820 (local single GPU: 0.830). ~55-65 min per run.
    # 23/11/2026: distribution strategies on the SAME hardware (T4), same global batch (16):
    # one T4 vs DDP on two, for ViT-S (control panda_ddp_512) and Swin-T (panda_ddp_swin_t); and
    # ViT-S with DDP and 2 data workers instead of 4 (is the 512-px ViT-S CPU-bound on Kaggle?).
    # Before the trainings the kernel measures the all-reduce bandwidth between the two T4.
    "strategies_512": [
        ("panda_single_vits", "configs/train_panda.yaml", single(model("vit_small_patch16_224")),
         "single"),
        ("panda_single_swin_t", "configs/train_panda.yaml",
         single(model("swin_tiny_patch4_window7_224")), "single"),
        ("panda_ddp_vits_w2", "configs/train_panda.yaml",
         deep_update(model("vit_small_patch16_224"), {"data": {"num_workers": 2}})),
    ],
}
BATCH = "strategies_512"
EXPERIMENTS = BATCHES[BATCH]
COMMON = {
    "data": {"num_workers": 4},
    "training": {"batch_size": COMMON_BATCH_PER_GPU},                  # per GPU -> global 16
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


STRATEGIES = ("ddp", "single")


def experiment_parts(exp: tuple) -> tuple[str, str, dict, str]:
    """``(name, base, overrides[, strategy])`` with the strategy defaulting to DDP on 2 GPUs."""
    name, base, overrides, *rest = exp
    strategy = rest[0] if rest else "ddp"
    if strategy not in STRATEGIES:
        raise ValueError(f"{name}: unknown strategy {strategy!r} (one of {STRATEGIES})")
    return name, base, overrides, strategy


def train_command(cfg: Path, strategy: str) -> tuple[list[str], dict | None]:
    """The training command and its extra environment (pure, tested): DDP on the 2 T4 with
    torchrun, or one process on the first T4 only."""
    common = ["--config", str(cfg), "--precision", "amp", "--monitor"]
    if strategy == "single":
        return ([sys.executable, "scripts/train_single_gpu.py", *common],
                {"CUDA_VISIBLE_DEVICES": "0"})
    return ([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
             "scripts/train_ddp.py", *common], None)


def best_checkpoint(name: str) -> Path | None:
    found = sorted(Path(f"/tmp/ckpt/{name}").rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    best = [p for p in found if p.name == "best.pt"]
    return (best or found or [None])[-1]


def copy_new_logs(src: Path, dst: Path, since: float) -> None:
    """Copy the files written by this kernel (mtime >= ``since``), not the committed history."""
    for f in src.rglob("*"):
        if f.is_file() and f.stat().st_mtime >= since:
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


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
    start = time.time()               # logs older than this came with the clone: not copied
    timings = []
    sh(sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
       "scripts/measure_allreduce.py", log=OUT / "allreduce.log")
    for exp in EXPERIMENTS:
        name, base, overrides, strategy = experiment_parts(exp)
        cfg = write_config(name, base, overrides)
        dest = OUT / "results" / name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg, dest / "config.yaml")
        t0 = time.time()
        cmd, extra_env = train_command(cfg, strategy)
        rc = sh(*cmd, log=dest / "train.log",
                env={**os.environ, **extra_env} if extra_env else None)
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
        timings.append({"name": name, "strategy": strategy, "train_rc": rc, "eval_rc": rc_eval,
                        "train_min": round(train_s / 60, 1), "checkpoint": ckpt.name if ckpt else ""})
        with open(OUT / "timings.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(timings[0]))
            writer.writeheader()
            writer.writerows(timings)
        copy_new_logs(REPO / "logs", OUT / "logs", since=start)
    print(open(OUT / "timings.csv").read(), flush=True)


if __name__ == "__main__":
    main()
