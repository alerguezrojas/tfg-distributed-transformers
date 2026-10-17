"""Download MedMNIST subsets (official .npz files from Zenodo, MD5-checked).

Usage:
    uv run python scripts/download_medmnist.py --root ~/datasets/medmnist
    uv run python scripts/download_medmnist.py --datasets medmnist_breast medmnist_derma --size 224

Then point a config at the folder: ``data: {name: medmnist_breast, root: <root>, size: 28}``.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.medmnist import DEFAULT_SIZE, SIZES, SUBSETS, download, npz_filename  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MedMNIST subsets")
    parser.add_argument("--root", required=True, help="Folder where the .npz files go")
    parser.add_argument("--datasets", nargs="+", choices=sorted(SUBSETS),
                        default=sorted(SUBSETS), help="Subsets to download (default: all)")
    parser.add_argument("--size", type=int, choices=SIZES, default=DEFAULT_SIZE,
                        help="Image size in pixels (default: 28)")
    args = parser.parse_args()

    for name in args.datasets:
        subset = SUBSETS[name]
        print(f"{name:18s} {npz_filename(subset, args.size):24s} ({subset.license}) ...",
              end=" ", flush=True)
        path = download(subset, args.root, args.size)
        print(f"ok — {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
