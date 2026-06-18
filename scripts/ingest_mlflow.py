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


def ingest_feasibility(client: MlflowClient) -> None:
    """The analytic feasibility model as its own MLflow experiment: one run per
    strategy scenario (speedup/time/VRAM), a DDP scaling curve, and predicted-vs-real."""
    try:
        from src.performance_model import predict
    except Exception:
        return
    GPU, MODEL = "Tesla T4", "vit_base_patch16_224"

    def P(strat, n, prec):
        return predict(strat, MODEL, GPU, n_gpus=n, dataset_size=5000, batch=96,
                       precision=prec, epochs=15)

    base = P("single", 1, "fp32")
    if base is None:
        return
    bt = base.time_per_epoch_train_s
    exp_id = client.create_experiment("Feasibility — analytic predictions",
                                      artifact_location=f"file:{ROOT / 'mlartifacts'}")

    for nm, strat, n, prec in [("Single · fp32", "single", 1, "fp32"),
                               ("DDP 2-GPU · fp32", "ddp", 2, "fp32"),
                               ("Single · AMP", "single", 1, "amp"),
                               ("DDP 2-GPU · AMP", "ddp", 2, "amp")]:
        p = P(strat, n, prec)
        rid = client.create_run(exp_id, run_name=nm,
                                tags={"strategy": strat, "precision": prec, "gpu": GPU,
                                      "bottleneck": p.bottleneck}).info.run_id
        for k, v in {"model": "vit_base", "strategy": strat, "n_gpus": n, "precision": prec, "gpu": GPU}.items():
            client.log_param(rid, k, str(v))
        client.log_metric(rid, "speedup", round(bt / p.time_per_epoch_train_s, 3))
        client.log_metric(rid, "time_per_epoch_s", round(p.time_per_epoch_train_s, 1))
        client.log_metric(rid, "vram_gb", round(p.vram_per_gpu_gb, 2))
        client.set_terminated(rid)

    rid = client.create_run(exp_id, run_name="DDP scaling (vit_base · T4)").info.run_id
    for n in (1, 2, 3, 4, 6, 8):
        client.log_metric(rid, "predicted_speedup", round(bt / P("ddp", n, "fp32").time_per_epoch_train_s, 3), step=n)
        client.log_metric(rid, "ideal_linear", float(n), step=n)
    client.set_terminated(rid)

    rid = client.create_run(exp_id, run_name="Predicted vs real (Kaggle 2×T4)").info.run_id
    client.log_metric(rid, "ddp_speedup_predicted", round(bt / P("ddp", 2, "fp32").time_per_epoch_train_s, 2))
    client.log_metric(rid, "ddp_speedup_real", 1.96)
    client.log_metric(rid, "amp_speedup_predicted", round(bt / P("single", 1, "amp").time_per_epoch_train_s, 2))
    client.log_metric(rid, "amp_speedup_real", 3.80)
    client.set_terminated(rid)
    print("  · Feasibility experiment: 4 scenarios + scaling curve + validation")


def ingest_dataset(client: MlflowClient) -> None:
    """The dataset as its own experiment: splits + per-class train counts + an
    imbalance bar chart artifact."""
    from src.web.dataset_stats import SPLIT_SIZES, class_distribution_approximate
    meta = next((Path(p) for p in (
        "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
        str(ROOT / "metadata.parquet")) if Path(p).exists()), None)
    dist = None
    if meta is not None:
        try:
            from src.web.dataset_stats import class_distribution_from_parquet
            dist = class_distribution_from_parquet(meta)
        except Exception:
            dist = None
    if dist is None:
        dist = class_distribution_approximate()
    dist = dist.sort_values("train_count", ascending=False)

    exp_id = client.create_experiment("Dataset — BigEarthNet-S2",
                                      artifact_location=f"file:{ROOT / 'mlartifacts'}")
    rid = client.create_run(exp_id, run_name="BigEarthNet-S2 (train split)").info.run_id
    for k, v in SPLIT_SIZES.items():
        client.log_param(rid, f"split_{k}", v)
    client.log_param(rid, "n_classes", len(dist))
    for _, row in dist.iterrows():
        nm = re.sub(r"[^0-9A-Za-z]+", "_", str(row["class"]))[:50].strip("_")
        client.log_metric(rid, f"count_{nm}", int(row["train_count"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = dist.sort_values("train_count")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(d["class"], d["train_count"], color="#3A536B")
    ax.set_title("BigEarthNet-S2 — class frequency in the train split (imbalance)")
    ax.set_xlabel("patches")
    fig.tight_layout()
    png = ROOT / "_dataset_bar.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    client.log_artifact(rid, str(png), "figures")
    png.unlink(missing_ok=True)
    client.set_terminated(rid)
    print("  · Dataset experiment: splits + class counts + imbalance chart")


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

    ingest_feasibility(client)
    ingest_dataset(client)

    print(f"\n✓ {n} runs ingested + Feasibility & Dataset experiments.")
    print("  Launch the UI with:  bash scripts/run_mlflow.sh")


if __name__ == "__main__":
    main()
