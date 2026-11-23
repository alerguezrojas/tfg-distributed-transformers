"""Kaggle (T4): training/eval throughput of many models with synthetic batches (no data, no I/O).

It runs ``scripts/benchmark.py`` (``paravit benchmark``) for each model of ``MODELS`` on ONE T4,
with the PANDA recipe (512 px, AMP) and the PANDA split sizes, so the only thing measured is
the GPU: this is the out-of-sample data for the performance predictor on a GPU it was not
calibrated on (the local calibration is on an RTX 3060 Ti). No dataset is attached. Output in
/kaggle/working:

    bench/<model>.log     the benchmark report of each model
    throughput.csv        model, batch, imgs/s train and eval, VRAM (GB) or OOM
    gpu.txt               nvidia-smi -q of the GPU (clocks, power limit)

Pushed with ``scripts/kaggle_push.sh kaggle/benchmark_t4`` (see ``kaggle/README.md``).
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/alerguezrojas/tfg-distributed-transformers.git"
REF = os.environ.get("PARAVIT_REF", "develop")
REPO = Path("/tmp/repo")
OUT = Path("/kaggle/working")
BATCHES = ["8", "16"]
TRAIN_SIZE, VAL_SIZE = 7433, 1593         # PANDA partition (splits.csv of the local runs)

# The 13 architectures of the PANDA model screening (docs/datasets/panda.md), 23/11/2026.
MODELS = [
    "vit_small_patch16_224", "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "eva02_small_patch14_224.mim_in22k", "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "swin_tiny_patch4_window7_224", "swinv2_tiny_window8_256", "maxvit_tiny_tf_512.in1k",
    "convnext_tiny.fb_in22k", "convnext_small.fb_in22k", "resnet50",
    "efficientnet_b0", "efficientnet_b3", "tf_efficientnetv2_s",
]

ROW = re.compile(r"^\s*(\d+)\s+off\s+(?:([\d.]+)\s+([\d.]+)\s+([\d.]+) GB|(OOM))", re.M)


def parse_rows(report: str) -> list[dict]:
    """``{batch, train_ips, eval_ips, vram_gb}`` of each ``trace=off`` row of a benchmark report
    (None values on OOM). Pure: tested in tests/unit/test_kaggle_kernels.py."""
    rows, seen = [], set()
    for m in ROW.finditer(report):
        if int(m.group(1)) in seen:         # the table can be printed twice (report + summary)
            continue
        seen.add(int(m.group(1)))
        oom = m.group(5) is not None
        rows.append({"batch": int(m.group(1)),
                     "train_ips": None if oom else float(m.group(2)),
                     "eval_ips": None if oom else float(m.group(3)),
                     "vram_gb": None if oom else float(m.group(4))})
    return rows


def sh(*cmd: str, log: Path | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    if log is None:
        return subprocess.run(cmd, check=False).returncode
    with open(log, "w") as f:
        rc = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT).returncode
    print(log.read_text()[-800:], flush=True)
    return rc


def main() -> None:
    sh("git", "clone", "--depth", "1", "--branch", REF, REPO_URL, str(REPO))
    sh(sys.executable, "-m", "pip", "install", "-q", "timm", "nvidia-ml-py", "torchinfo")
    with open(OUT / "gpu.txt", "w") as f:
        subprocess.run(["nvidia-smi", "-q"], check=False, stdout=f)
    os.chdir(REPO)
    bench = OUT / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    out = []
    for model in MODELS:
        log = bench / f"{model}.log"
        rc = sh(sys.executable, "scripts/benchmark.py", "--model", model, "--batch-sizes", *BATCHES,
                "--precision", "amp", "--config", "configs/train_panda.yaml", "--epochs", "15",
                "--trace-modes", "off", "--train-size", str(TRAIN_SIZE), "--val-size", str(VAL_SIZE),
                "--device", "0", log=log)
        rows = parse_rows(log.read_text()) or [{"batch": None, "train_ips": None,
                                                "eval_ips": None, "vram_gb": None}]
        out += [{"model": model, "rc": rc, **r} for r in rows]
        with open(OUT / "throughput.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    print(open(OUT / "throughput.csv").read(), flush=True)


if __name__ == "__main__":
    main()
