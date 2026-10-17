"""Mean ± std of the held-out test metrics over repeated (seeded) runs.

Usage:
    uv run python scripts/summarize_runs.py logs/local/medmnist_*/single/*

Prints one Markdown row per run folder: number of runs, seeds and every test metric
as mean ± sample standard deviation.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.results_summary import METRICS, collect, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate test results over seeds")
    parser.add_argument("folders", nargs="+", type=Path, help="Run folders (logs/...)")
    args = parser.parse_args()

    print("| Run folder | n | seeds | " + " | ".join(METRICS) + " |")
    print("|---|---|---|" + "---|" * len(METRICS))
    for folder in args.folders:
        runs = collect(folder)
        if not runs:
            continue
        stats = summarize(runs)
        cells = [f"{stats[k][0]:.4f} ± {stats[k][1]:.4f}" if k in stats else "—"
                 for k in METRICS]
        seeds = ",".join(str(r.seed) for r in runs)
        print(f"| {folder} | {len(runs)} | {seeds} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
