"""Data layer for the Dash dashboard — loads runs / dataset / feasibility once.

Pure Python (reuses src.web.run_registry + the CSV parsers), no Dash imports, so
it stays testable and is the single source of truth for every view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web.run_registry import discover_runs  # noqa: E402

_MODE_LABEL = {"single": "Single-GPU", "ddp": "DDP", "ddp_hetero": "Heterogeneous",
               "model_parallel": "Model-parallel"}


def _read_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _curve(epoch_csv) -> dict:
    if not epoch_csv:
        return {}
    df = _read_csv(epoch_csv)
    if df.empty or "epoch" not in df.columns:
        return {}
    cols = ["epoch", "train_f1", "val_f1", "train_loss", "val_loss",
            "train_acc", "val_acc", "val_prec", "val_rec", "epoch_time_s"]
    return {c: [None if pd.isna(v) else round(float(v), 5) for v in df[c]]
            for c in cols if c in df.columns}


def _perclass(perclass_csv) -> list[dict]:
    if not perclass_csv:
        return []
    df = _read_csv(perclass_csv)
    if df.empty or "epoch" not in df.columns:
        return []
    df = df[df["epoch"] == df["epoch"].max()]
    return [{"cls": str(r.get("class_name", "")), "f1": round(float(r.get("f1", 0)), 4),
             "precision": round(float(r.get("precision", 0)), 4),
             "recall": round(float(r.get("recall", 0)), 4)} for _, r in df.iterrows()]


def _date(ts: str) -> str:
    if len(ts) >= 13 and int(ts[4:8]) >= 2000:
        return f"{ts[:2]}/{ts[2:4]}/{ts[4:8]} {ts[9:11]}:{ts[11:13]}"
    return f"{ts[6:8]}/{ts[4:6]}/{ts[:4]} {ts[9:11]}:{ts[11:13]}"


def load_runs() -> list[dict]:
    out = []
    for r in discover_runs(ROOT):
        curve = _curve(r.epoch_csv_path)
        vals = [v for v in (curve.get("val_f1") or []) if v is not None]
        best = max(vals) if vals else None
        all_vf = curve.get("val_f1") or []
        best_ep = (all_vf.index(best) + 1) if best is not None else None
        times = [t for t in (curve.get("epoch_time_s") or []) if t]
        out.append({
            "id": r.timestamp, "label": r.label, "env": r.env, "mode": r.mode,
            "mode_label": _MODE_LABEL.get(r.mode, r.mode),
            "model": r.model.replace("_patch16_224", "") or "—",
            "precision": r.precision or "fp32", "trace": r.trace_mode, "date": _date(r.timestamp),
            "epochs": len(all_vf), "best_f1": best, "best_epoch": best_ep,
            "duration_min": round(sum(times) / 60, 1) if times else None,
            "curve": curve, "perclass": _perclass(r.perclass_csv_path),
        })
    return out


def load_dataset() -> dict:
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
    classes = [{"cls": str(r["class"]), "count": int(r["train_count"])}
               for _, r in dist.sort_values("train_count", ascending=False).iterrows()]
    return {"splits": dict(SPLIT_SIZES), "classes": classes}


def load_feasibility() -> dict:
    try:
        from src.performance_model import predict
    except Exception:
        return {}
    GPU, MODEL = "Tesla T4", "vit_base_patch16_224"

    def P(strat, n, prec):
        return predict(strat, MODEL, GPU, n_gpus=n, dataset_size=5000, batch=96,
                       precision=prec, epochs=15)

    base = P("single", 1, "fp32")
    if base is None:
        return {}
    bt = base.time_per_epoch_train_s
    sp = lambda p: round(bt / p.time_per_epoch_train_s, 2) if p and p.time_per_epoch_train_s else None
    scen = [("Single", "single", 1, "fp32"), ("DDP · 2 GPU", "ddp", 2, "fp32"),
            ("Single · AMP", "single", 1, "amp"), ("DDP · 2 GPU · AMP", "ddp", 2, "amp")]
    rows = []
    for name, s, n, prec in scen:
        p = P(s, n, prec)
        if p:
            rows.append({"name": name, "prec": prec, "gpus": n,
                         "time": round(p.time_per_epoch_train_s, 0), "speedup": sp(p),
                         "vram": round(p.vram_per_gpu_gb, 1), "bottleneck": p.bottleneck})
    scaling = [{"n": n, "speedup": sp(P("ddp", n, "fp32"))} for n in (1, 2, 3, 4, 6, 8)]
    big = predict("single", "vit_large_patch16_224", GPU, n_gpus=1, batch=48, precision="fp32")
    validation = [
        {"q": "DDP 2×T4 speedup", "pred": f"{sp(P('ddp', 2, 'fp32')):.2f}×", "real": "1.96×"},
        {"q": "FP32 → AMP speedup", "pred": f"{sp(P('single', 1, 'amp')):.2f}×", "real": "3.80×"},
        {"q": "vit_large @ batch 48 (1 T4)", "pred": "OOM" if big and not big.fits_in_memory else "fits", "real": "OOM"},
    ]
    return {"gpu": GPU, "model": "vit_base", "scenarios": rows, "scaling": scaling, "validation": validation}


# Loaded once at import (the dashboard is read-only over a fixed logs/ tree).
RUNS = load_runs()
BY_ID = {r["id"]: r for r in RUNS}
DATASET = load_dataset()
FEAS = load_feasibility()
