"""Ingest the existing training runs (logs/ CSVs) into a local MLflow store.

Our runs are already trained and saved as logs + CSVs. This replays them into
MLflow's tracking store (./mlruns) — params, per-epoch metrics (with step), best
F1, per-class F1 and the artifact CSVs — so MLflow's own polished UI shows them:

    .venv-mlflow/bin/python scripts/ingest_mlflow.py     # build ./mlruns
    .venv-mlflow/bin/mlflow ui                            # → http://127.0.0.1:5000

Runs in the isolated .venv-mlflow environment (keeps MLflow's deps off the main
project). Reuses the pure run_registry; reads CSVs with pandas.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.web.run_registry import discover_runs  # noqa: E402

EXPERIMENT = "BigEarthNet-S2 · Distributed Transformers"
_MODE_LABEL = {"single": "Single-GPU", "ddp": "DDP", "ddp_hetero": "Heterogeneous",
               "model_parallel": "Model-parallel"}
_EPOCH_METRICS = ["train_f1", "val_f1", "train_loss", "val_loss",
                  "train_acc", "val_acc", "val_prec", "val_rec"]


def _config_line(log_path: Path) -> dict:
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            i = line.find("Configuración:")
            if i < 0:
                continue
            out = {}
            for part in line[i + len("Configuración:"):].split("|"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k.strip()] = v.strip()
            return out
    except OSError:
        pass
    return {}


def _start_ms(sort_key: str) -> int:
    try:
        return int(datetime.strptime(sort_key, "%Y%m%d_%H%M%S").timestamp() * 1000)
    except ValueError:
        return int(datetime.now().timestamp() * 1000)


def main() -> None:
    # MLflow 3.x retired the file store → use a local sqlite backend + artifact dir.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = (exp.experiment_id if exp else
              client.create_experiment(EXPERIMENT, artifact_location=f"file:{ROOT / 'mlartifacts'}"))

    runs = discover_runs(ROOT)
    n = 0
    for r in runs:
        cfg = _config_line(r.log_path)
        epoch_df = pd.read_csv(r.epoch_csv_path) if r.epoch_csv_path else pd.DataFrame()

        start = _start_ms(r.sort_key)
        tags = {"mlflow.runName": r.label, "environment": r.env,
                "strategy": _MODE_LABEL.get(r.mode, r.mode), "model": r.model or "—",
                "precision": r.precision or "fp32", "trace": r.trace_mode}
        run = client.create_run(exp_id, start_time=start, tags=tags)
        rid = run.info.run_id

        # ── Params (hyperparameters) ──────────────────────────────────────────
        params = {"model": r.model or "—", "strategy": r.mode,
                  "precision": r.precision or "fp32", "environment": r.env}
        bm = re.search(r"\d+", cfg.get("batch", ""))
        if bm:
            params["batch_size"] = bm.group()
        if cfg.get("lr"):
            params["lr"] = cfg["lr"]
        if cfg.get("loss"):
            params["loss"] = cfg["loss"]
        if not epoch_df.empty:
            params["epochs"] = str(len(epoch_df))
        for k, v in params.items():
            client.log_param(rid, k, v)

        # ── Per-epoch metrics (step = epoch) ──────────────────────────────────
        best = None
        if not epoch_df.empty and "epoch" in epoch_df.columns:
            for _, row in epoch_df.iterrows():
                step = int(row["epoch"])
                for m in _EPOCH_METRICS:
                    if m in epoch_df.columns and pd.notna(row[m]):
                        client.log_metric(rid, m, float(row[m]), step=step)
            if "val_f1" in epoch_df.columns and epoch_df["val_f1"].notna().any():
                best = float(epoch_df["val_f1"].max())
                client.log_metric(rid, "best_val_f1", best)

        # ── Per-class F1 (final epoch) as metrics + the CSV artifacts ─────────
        if r.perclass_csv_path and Path(r.perclass_csv_path).exists():
            pc = pd.read_csv(r.perclass_csv_path)
            if not pc.empty and "epoch" in pc.columns:
                last = pc[pc["epoch"] == pc["epoch"].max()]
                for _, row in last.iterrows():
                    name = re.sub(r"[^0-9A-Za-z]+", "_", str(row.get("class_name", "")))[:60]
                    if name and pd.notna(row.get("f1")):
                        client.log_metric(rid, f"f1_{name}", float(row["f1"]))
            client.log_artifact(rid, str(r.perclass_csv_path), "per_class")
        for p in (r.epoch_csv_path, r.confusion_matrix_csv_path, r.batch_csv_path, r.log_path):
            if p and Path(p).exists():
                client.log_artifact(rid, str(p))

        client.set_terminated(rid, end_time=start + 1000)
        n += 1
        print(f"  · {r.label}" + (f"  (best F1 {best:.3f})" if best else ""))

    print(f"\n✓ {n} runs ingested into ./mlruns (experiment '{EXPERIMENT}').")
    print("  Launch the UI with:  .venv-mlflow/bin/mlflow ui")


if __name__ == "__main__":
    main()
