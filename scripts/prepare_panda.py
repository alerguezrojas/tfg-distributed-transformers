"""Prepare PANDA offline: one tissue mosaic per biopsy and the fixed train/val/test split.

For each biopsy of ``train.csv`` it reads pyramid level ``--level`` of its TIFF, cuts it
into ``--tile`` px tiles, keeps the ``--n-tiles`` with the most tissue and saves them as a
square mosaic (4x4 tiles of 256 px = 1024 px by default) in ``--out`` as JPEG quality 95
without chroma subsampling: ~2.7x smaller than PNG on H&E images (0.7 vs 2.0 MB) with no
visible loss, so the 10.6 k mosaics take ~8 GB. Already-done mosaics are skipped, unreadable
TIFFs are logged in ``<out>/errors.csv`` and the run goes on.

Usage:
    uv run python scripts/prepare_panda.py --root /media/alejandro/SSD/datasets/panda
    uv run python scripts/prepare_panda.py --root <root> --workers 8 --make-splits
    uv run python scripts/prepare_panda.py --root <root> --no-mosaics --make-splits   # splits only

``--make-splits`` writes ``<root>/splits.csv`` (70/15/15 stratified by centre and ISUP
grade, seed 42, per biopsy: PANDA has no patient id), leaving out the ids in errors.csv.
Then point a config at ``<root>`` (``data: {name: panda, root: <root>}``).
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.wsi import grid_side, make_mosaic, read_level, tissue_fraction  # noqa: E402

JPEG_QUALITY = 95


def mosaic_name(level: int, tile: int, n_tiles: int) -> str:
    return f"mosaics_l{level}_{n_tiles}x{tile}"


def process_one(job: tuple) -> tuple[str, str, str, float, float]:
    """Make the mosaic of one biopsy → (id, status, message, seconds, tissue fraction).
    status: 'done' | 'skipped' | 'error'. Never raises (one bad TIFF must not stop a run)."""
    image_id, tiff, out_dir, level, tile, n_tiles = job
    dest = Path(out_dir) / f"{image_id}.jpg"
    if dest.exists():
        return image_id, "skipped", "", 0.0, float("nan")
    t0 = time.time()
    try:
        img = read_level(tiff, level)
        mosaic = make_mosaic(img, tile, n_tiles)
        tmp = dest.with_suffix(".tmp")      # atomic: a killed run leaves no half-written file
        Image.fromarray(mosaic).save(tmp, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
        tmp.replace(dest)
        return image_id, "done", "", time.time() - t0, tissue_fraction(mosaic)
    except Exception as exc:    # unreadable / missing / truncated TIFF: log and go on
        return image_id, "error", f"{type(exc).__name__}: {exc}", time.time() - t0, float("nan")


def build_mosaics(args, labels: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ids = labels["image_id"].tolist()[: args.limit] if args.limit else labels["image_id"].tolist()
    jobs = [(i, str(args.root / "train_images" / f"{i}.tiff"), str(out),
             args.level, args.tile, args.n_tiles) for i in ids]
    errors, done, times, tissue = [], 0, [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for iid, status, msg, secs, frac in tqdm(pool.map(process_one, jobs, chunksize=4),
                                                 total=len(jobs), desc="mosaics", unit="img"):
            if status == "error":
                errors.append((iid, msg))
            elif status == "done":
                done += 1
                times.append(secs)
                tissue.append(frac)
    with open(out / "errors.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "error"])
        writer.writerows(errors)
    mean_t = f"{np.mean(times):.2f} s/biopsy, tissue {np.mean(tissue):.0%}" if times else "-"
    print(f"{done} new, {len(jobs) - done - len(errors)} already there, "
          f"{len(errors)} errors (see {out / 'errors.csv'}); {mean_t}")


def write_splits(args, labels: pd.DataFrame, out: Path) -> None:
    from src.data.panda import SPLIT_SEED, make_splits   # pulls torch: only when needed
    bad = out / "errors.csv"
    if bad.exists():
        skip = set(pd.read_csv(bad)["image_id"]) if bad.stat().st_size else set()
        labels = labels[~labels["image_id"].isin(skip)]
        if skip:
            print(f"splits: leaving out {len(skip)} ids from {bad}")
    splits = make_splits(labels, seed=SPLIT_SEED)
    path = args.root / "splits.csv"
    splits.to_csv(path, index=False)
    print(f"{path}: {splits['split'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PANDA: tissue mosaics and fixed splits")
    parser.add_argument("--root", required=True, type=Path,
                        help="Folder with train.csv and train_images/")
    parser.add_argument("--out", type=Path, default=None,
                        help="Mosaic folder (default: <root>/mosaics_l<level>_<n>x<tile>)")
    parser.add_argument("--level", type=int, default=1, help="Pyramid level (1 = 4x reduced)")
    parser.add_argument("--tile", type=int, default=256, help="Tile side in pixels")
    parser.add_argument("--n-tiles", type=int, default=16,
                        help="Tiles per mosaic (perfect square: 16 → 4x4)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel processes")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N biopsies (test)")
    parser.add_argument("--no-mosaics", action="store_true", help="Skip the mosaics")
    parser.add_argument("--make-splits", action="store_true",
                        help="Also write <root>/splits.csv (70/15/15, seed 42)")
    args = parser.parse_args()
    grid_side(args.n_tiles)
    args.root = args.root.expanduser()
    out = (args.out or args.root / mosaic_name(args.level, args.tile, args.n_tiles)).expanduser()
    labels = pd.read_csv(args.root / "train.csv")

    if not args.no_mosaics:
        build_mosaics(args, labels, out)
    if args.make_splits:
        write_splits(args, labels, out)


if __name__ == "__main__":
    main()
