"""Throughput sweep of the performance predictor on the GPU of this machine (cluster, new PC).

The same measurement campaigns as the T4 kernel (``kaggle/benchmark_t4/run.py``: synthetic
batches, no data and no I/O, PANDA split sizes), run from this checkout and with this Python, plus
the AMP campaign at 512 px of the first T4 kernel (23/11/2026), so every GPU has the same table.
Measurements of a GPU the predictor has not been calibrated on are its out-of-sample validation.

    python cluster/benchmark/throughput_sweep.py --out ~/paravit_runs/v100_sweep
    python cluster/benchmark/throughput_sweep.py --out DIR --campaigns amp_224_lotes

Output in ``--out``: ``bench/<campaign>/<model>.log``, ``throughput.csv`` (rewritten after each
model) and ``gpu.txt`` (``nvidia-smi -q``). In VERODE it runs on the compute node with ``srun``
(see ``cluster/README.md``).
"""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _kernel():
    spec = importlib.util.spec_from_file_location("benchmark_t4_run",
                                                  ROOT / "kaggle" / "benchmark_t4" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def campaigns(kernel=None) -> list[tuple]:
    """The T4 campaigns, with the AMP one at 512 px (batches 8 and 16, the 13 models) first."""
    k = kernel or _kernel()
    return [("amp_512", 512, "amp", ["8", "16"], k.MODELS), *k.CAMPAIGNS]


def main(argv: list[str] | None = None) -> None:
    k = _kernel()
    all_campaigns = campaigns(k)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--campaigns", nargs="+", choices=[c[0] for c in all_campaigns],
                        help="subset of campaigns (default: all)")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    chosen = [c for c in all_campaigns if args.campaigns is None or c[0] in args.campaigns]
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gpu.txt", "w") as f:
        subprocess.run(["nvidia-smi", "-q"], check=False, stdout=f)
    os.chdir(ROOT)
    k.run_campaigns(out, chosen, python=sys.executable, device=args.device)
    print((out / "throughput.csv").read_text(), flush=True)


if __name__ == "__main__":
    main()
