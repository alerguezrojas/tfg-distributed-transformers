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
                  "train_acc", "val_acc", "val_prec", "val_rec", "epoch_time_s"]


def _parse_log_series(log_path: Path) -> dict[int, dict]:
    """Per-epoch series the CSV doesn't hold: F1 at the optimal threshold, the
    threshold itself, and train/eval energy (Wh) + power (W) — parsed from the log."""
    out: dict[int, dict] = {}
    cur = None
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        m = re.search(r"Epoch (\d+)/", line)
        if m:
            cur = int(m.group(1)); out.setdefault(cur, {}); continue
        if cur is None:
            continue
        e = re.search(r"\[energy\] \w+\.(train|eval)_epoch:.*?\(([\d.]+) Wh\).*?media ([\d.]+) W", line)
        if e:
            out[cur][f"energy_{e.group(1)}_wh"] = float(e.group(2))
            out[cur][f"power_{e.group(1)}_w"] = float(e.group(3))
            continue
        t = re.search(r"threshold óptimo=([\d.]+), F1=([\d.]+)", line)
        if t:
            out[cur]["optimal_threshold"] = float(t.group(1))
            out[cur]["f1_at_optimal_threshold"] = float(t.group(2))
    return out


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
    # The store is fully derived from logs/, so rebuild it from scratch each run →
    # idempotent (no duplicated runs when re-ingesting after new trainings).
    import shutil
    (ROOT / "mlflow.db").unlink(missing_ok=True)
    shutil.rmtree(ROOT / "mlartifacts", ignore_errors=True)

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
                client.log_param(rid, "best_epoch", int(epoch_df.loc[epoch_df["val_f1"].idxmax(), "epoch"]))
            if "epoch_time_s" in epoch_df.columns and epoch_df["epoch_time_s"].notna().any():
                client.log_metric(rid, "total_duration_min", round(epoch_df["epoch_time_s"].dropna().sum() / 60, 2))

        # ── Per-epoch series only in the log: F1@optimal-threshold, energy, power ─
        for step, vals in _parse_log_series(r.log_path).items():
            for k, v in vals.items():
                client.log_metric(rid, k, float(v), step=step)

        # ── Per-class F1 (final epoch) as metrics + the CSV artifacts ─────────
        if r.perclass_csv_path and Path(r.perclass_csv_path).exists():
            pc = pd.read_csv(r.perclass_csv_path)
            if not pc.empty and "epoch" in pc.columns:
                last = pc[pc["epoch"] == pc["epoch"].max()]
                for _, row in last.iterrows():
                    name = re.sub(r"[^0-9A-Za-z]+", "_", str(row.get("class_name", "")))[:50].strip("_")
                    if not name:
                        continue
                    for col, prefix in (("f1", "f1"), ("precision", "prec"), ("recall", "recall")):
                        if col in last.columns and pd.notna(row.get(col)):
                            client.log_metric(rid, f"{prefix}_{name}", float(row[col]))
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
