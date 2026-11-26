"""PANDA: how much tissue a mosaic of N tiles leaves out (does a bigger mosaic see more biopsy?).

Reads the mosaics of ``scripts/prepare_panda.py`` with many tiles (e.g. 36, ``mosaics_l1_36x256``).
Their tiles are pasted row by row in the order ``src/data/wsi.select_tiles`` chose them (darkest
first), so the first ``--keep`` tiles are exactly the mosaic of ``--keep`` tiles of the same level
and tile size. For every biopsy it counts the tiles with tissue and the share of the mosaic's
tissue those first ``--keep`` tiles hold. Summary overall, by centre and by ISUP grade.

    uv run python scripts/panda_tissue_coverage.py --root /media/alejandro/SSD/datasets/panda \\
        --mosaics mosaics_l1_36x256 --keep 16

Tissue = pixels darker than ``--threshold`` in grey levels (the background of an H&E slide is
near white); a tile "has tissue" when at least ``--min-fraction`` of it is tissue. A rough
measure, good to compare tile counts, not to segment.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def tile_tissue_fractions(gray: np.ndarray, n_tiles: int, threshold: int = 210) -> np.ndarray:
    """Fraction of tissue pixels of each tile of a square mosaic (row-major order). Pure."""
    side = math.isqrt(n_tiles)
    if side * side != n_tiles:
        raise ValueError(f"n_tiles must be a perfect square, got {n_tiles}")
    t = gray.shape[0] // side
    tissue = gray < threshold
    return np.array([tissue[i * t:(i + 1) * t, j * t:(j + 1) * t].mean()
                     for i in range(side) for j in range(side)])


def coverage(fractions: np.ndarray, keep: int, min_fraction: float = 0.25) -> tuple[int, float]:
    """``(tiles with tissue, share of the tissue held by the first ``keep`` tiles)``, with the
    tiles in mosaic order (the order the smaller mosaic keeps them). Pure."""
    total = float(fractions.sum())
    kept = float(fractions[:keep].sum())
    return int((fractions >= min_fraction).sum()), (kept / total if total > 0 else 1.0)


def _one(path: Path, n_tiles: int, keep: int, threshold: int, min_fraction: float):
    from PIL import Image
    gray = np.asarray(Image.open(path).convert("L").reduce(4), dtype=np.uint8)
    return (path.stem, *coverage(tile_tissue_fractions(gray, n_tiles, threshold), keep, min_fraction))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True, help="folder with train.csv")
    p.add_argument("--mosaics", default="mosaics_l1_36x256", help="mosaic folder under --root")
    p.add_argument("--keep", type=int, default=16, help="tiles of the smaller mosaic to compare")
    p.add_argument("--threshold", type=int, default=210)
    p.add_argument("--min-fraction", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    folder = args.root / args.mosaics
    files = sorted(folder.glob("*.jpg"))
    n_tiles = int(args.mosaics.split("_")[-1].split("x")[0])
    work = partial(_one, n_tiles=n_tiles, keep=args.keep, threshold=args.threshold,
                   min_fraction=args.min_fraction)
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(work, files, chunksize=64))
    with open(args.root / "train.csv", newline="") as f:
        meta = {r["image_id"]: r for r in csv.DictReader(f)}
    n = np.array([r[1] for r in rows])
    lost = 1 - np.array([r[2] for r in rows])
    print(f"{len(rows)} biopsias, mosaicos de {n_tiles} tiles ({folder.name}); "
          f"comparación con sus {args.keep} primeros tiles (el mosaico de {args.keep})\n")
    print(f"Tiles con tejido: mediana {np.median(n):.0f} (p25 {np.percentile(n, 25):.0f}, "
          f"p75 {np.percentile(n, 75):.0f}); más de {args.keep}: {100 * np.mean(n > args.keep):.1f} %")
    print(f"Al menos 30 tiles con tejido: {100 * np.mean(n >= 30):.1f} %")
    print(f"Tejido que dejan fuera {args.keep} tiles: media {100 * lost.mean():.1f} %, "
          f"mediana {100 * np.median(lost):.1f} %\n")
    groups = [(c, "data_provider", c) for c in ("karolinska", "radboud")]
    groups += [(f"ISUP {g}", "isup_grade", str(g)) for g in range(6)]
    for label, key, value in groups:
        m = np.array([meta[r[0]][key] == value for r in rows])
        print(f"  {label:11s} n={m.sum():5d}  tiles con tejido {np.median(n[m]):4.0f}  "
              f"tejido fuera {100 * lost[m].mean():5.1f} %")


if __name__ == "__main__":
    main()
