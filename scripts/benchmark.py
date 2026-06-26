"""Pre-training benchmark analysis — thin CLI.

The analysis logic lives in the ``src/benchmark`` package (one responsibility
per module: value objects, model analyzer, probes, predictor, DDP optimizer,
benchmarker, time estimator, report formatter, and the BenchmarkChecker
facade). This file only parses arguments and wires the facade to the formatter.

Examples:
    uv run python scripts/benchmark.py --batch-sizes 16 32 --epochs 30
    uv run python scripts/benchmark.py --model resnet50 --batch-sizes 32 64
    uv run python scripts/benchmark.py --config configs/train_cluster_v3.yaml \\
        --batch-sizes 64 --nfs-factor 1.3 --convergence-study
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import BenchmarkChecker, ReportFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analizador de viabilidad pre-entrenamiento — BigEarthNet ViT v3"
    )
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Modelo(s) timm (separados por espacio)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    parser.add_argument("--trace-modes", nargs="+",
                        choices=["off", "simple", "deep"], default=["off", "simple"])
    parser.add_argument("--nfs-factor", type=float, default=1.0, metavar="FACTOR",
                        help="Multiplicador de overhead para almacenamiento NFS (p.ej. 1.3 para Verode)")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Ruta al dataset BigEarthNet-S2 para medir I/O real")
    parser.add_argument("--no-disk-profile", action="store_true",
                        help="Omitir medición de I/O del disco (más rápido)")
    parser.add_argument("--no-ddp-analysis", action="store_true",
                        help="Omitir análisis DDP")
    parser.add_argument("--no-prediction", action="store_true",
                        help="Omitir predicción de rendimiento F1")
    parser.add_argument("--convergence-study", action="store_true",
                        help="Ejecuta un mini-training REAL: LR range test + curva de "
                             "convergencia medida + gradient noise scale (más lento, ~3-8 min)")
    parser.add_argument("--study-steps", type=int, default=60,
                        help="Número de steps del mini-training de convergencia (default 60)")
    parser.add_argument("--device", type=int, default=0, metavar="INDEX",
                        help="Índice de GPU CUDA a usar (default 0). Útil en máquinas multi-GPU.")
    parser.add_argument("--precision", choices=["fp32", "tf32", "amp", "bf16"], default="fp32",
                        help="Precisión del benchmark = interruptor de Tensor cores "
                             "(fp32=CUDA cores; tf32/amp/bf16=Tensor cores).")
    parser.add_argument("--compare-precision", action="store_true",
                        help="Mide FP32 vs la mejor precisión Tensor-core y reporta el speedup.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Heterogeneous configs put batch_size per-rank under distributed.ranks and omit
    # the top-level training.batch_size, so fall back to the largest per-rank batch.
    default_bs = cfg["training"].get("batch_size")
    if default_bs is None:
        ranks = (cfg.get("distributed") or {}).get("ranks") or []
        per_rank = [r.get("batch_size") for r in ranks if r.get("batch_size")]
        default_bs = max(per_rank) if per_rank else 32
    batch_sizes = args.batch_sizes or [default_bs]
    epochs_list = args.epochs or [cfg["training"]["epochs"]]
    model_names = args.model or [cfg["model"]["name"]]
    env = cfg.get("output", {}).get("env", "local")

    # Auto-detect dataset path from config if not provided
    dataset_path = args.dataset_path or cfg.get("data", {}).get("root")

    # Tamaño REAL del dataset según el metadata del config — NO asumir el full
    # set. Si el config apunta a un subset (p.ej. metadata_demo.parquet con 5000
    # imágenes), las estimaciones deben usar ese tamaño para que sean comparables
    # con el run real. Fallback al full BigEarthNet si no se puede leer.
    n_train, n_val = 237871, 122342
    meta_path = cfg.get("data", {}).get("metadata")
    if meta_path and Path(meta_path).exists():
        try:
            import pandas as pd
            counts = pd.read_parquet(meta_path, columns=["split"])["split"].value_counts()
            n_train = int(counts.get("train", n_train))
            n_val = int(counts.get("validation", n_val))
            print(f"Dataset (metadata): train={n_train:,}  val={n_val:,}")
        except Exception as exc:
            print(f"[aviso] no se pudo leer el tamaño del metadata ({exc}); "
                  f"usando full set {n_train:,}/{n_val:,}")

    for model_name in model_names:
        output_path = args.output
        if output_path is None:
            ts = datetime.now().strftime("%d%m%Y_%H%M%S")
            output_path = Path(f"logs/{env}/benchmark/benchmark_{ts}.log")

        checker = BenchmarkChecker(
            model_name=model_name,
            batch_sizes=batch_sizes,
            epochs_list=epochs_list,
            trace_modes=args.trace_modes,
            dataset_train=n_train,
            dataset_val=n_val,
            nfs_factor=args.nfs_factor,
            dataset_path=dataset_path,
            profile_disk=not args.no_disk_profile,
            predict_performance=not args.no_prediction,
            analyze_ddp=not args.no_ddp_analysis,
            config=cfg,
            convergence_study=args.convergence_study,
            study_steps=args.study_steps,
            device_index=args.device,
            precision=args.precision,
            compare_precision=args.compare_precision,
        )

        report = checker.run()
        formatter = ReportFormatter(output_path=output_path)
        formatter.print(report)
        formatter.write_csv(report, env=env)
        output_path = None  # reset para el siguiente modelo


if __name__ == "__main__":
    main()
