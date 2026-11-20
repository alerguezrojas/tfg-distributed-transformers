"""Kaggle (CPU): PANDA tissue mosaics and fixed splits, straight from the competition data.

Same recipe as the local preparation (``scripts/prepare_panda.py``: level 1, 16 tiles of 256 px,
1024 px JPEG mosaics) so the runs on Kaggle and on the local GPU see identical inputs. The
competition folder is read-only, so the root is rebuilt in /tmp with links, and only the mosaics,
``splits.csv`` and ``train.csv`` go to /kaggle/working (the kernel output, reused by the training
kernels). The md5 of ``splits.csv`` is printed to check it against the local one.

Pushed with ``scripts/kaggle_push.sh kaggle/panda_mosaicos`` (see ``kaggle/README.md``).
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/alerguezrojas/tfg-distributed-transformers.git"
REF = os.environ.get("PARAVIT_REF", "develop")
ROOT = Path("/tmp/panda")             # writable root: links to the read-only competition data
OUT = Path("/kaggle/working")
MOSAICS = "mosaics_l1_16x256"


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_competition() -> Path:
    """The competition mount point has changed over time; find the folder with the slides."""
    for csv in sorted(Path("/kaggle/input").rglob("train.csv")):
        if (csv.parent / "train_images").is_dir():
            return csv.parent
    raise SystemExit("PANDA competition data not found under /kaggle/input")


def main() -> None:
    comp = find_competition()
    print(f"competition data: {comp}", flush=True)
    sh("git", "clone", "--depth", "1", "--branch", REF, REPO_URL, "/tmp/repo")
    sh(sys.executable, "-m", "pip", "install", "-q", "tifffile", "imagecodecs", "timm")
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy(comp / "train.csv", ROOT / "train.csv")
    (ROOT / "train_images").symlink_to(comp / "train_images")
    sh(sys.executable, "/tmp/repo/scripts/prepare_panda.py", "--root", str(ROOT),
       "--out", str(OUT / MOSAICS), "--workers", str(os.cpu_count() or 4), "--make-splits")
    for name in ("splits.csv", "train.csv"):
        shutil.copy(ROOT / name, OUT / name)
    digest = hashlib.md5((OUT / "splits.csv").read_bytes()).hexdigest()
    n = sum(1 for _ in (OUT / MOSAICS).glob("*.jpg"))
    print(f"splits.csv md5 {digest} | {n} mosaics in {OUT / MOSAICS}", flush=True)


if __name__ == "__main__":
    main()
