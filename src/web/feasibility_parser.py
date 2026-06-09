"""Parse feasibility CSV generado por check_feasibility.py v3.

Maneja el formato legacy (v1/v2) y el nuevo (v3) con bloques #cpu, #disk,
#dataset, #prediction, #curve_val_f1, #curve_train_f1, #ddp.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def parse_feasibility_csv(csv_path: Path) -> tuple[dict, pd.DataFrame]:
    """Reads the feasibility CSV and returns (metadata_dict, benchmark_df).

    The metadata_dict includes every block: #meta, #model_mem, #cpu, #disk,
    #dataset, #prediction, #curve_val_f1, #curve_train_f1, #ddp.
    The benchmark_df holds the benchmark data rows.
    """
    rows: list[list[str]] = []
    meta: dict = {}
    sizes: dict = {}
    model_mem: dict = {}
    cpu_info: dict = {}
    disk_info: dict = {}
    dataset_info: dict = {}
    prediction: dict = {}
    ddp_rows: list[dict] = []
    curve_val: list[float] = []
    curve_train: list[float] = []
    curve_epochs: list[int] = []
    # Empirical convergence study (v4)
    study: dict = {}

    def _floats(seq):
        out = []
        for v in seq:
            if v == "":
                continue
            try:
                out.append(float(v))
            except ValueError:
                pass
        return out

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header_meta: list[str] = []
        header_sizes: list[str] = []
        header_model_mem: list[str] = []
        header_cpu: list[str] = []
        header_disk: list[str] = []
        header_dataset: list[str] = []
        header_prediction: list[str] = []
        header_ddp: list[str] = []
        header_study_lr: list[str] = []
        header_study_conv: list[str] = []
        header_study_grad: list[str] = []

        for row in reader:
            if not row:
                continue

            tag = row[0]

            if tag == "#meta":
                if not header_meta:
                    header_meta = row[1:]
                else:
                    meta = dict(zip(header_meta, row[1:]))

            elif tag == "#sizes":
                if not header_sizes:
                    header_sizes = row[1:]
                else:
                    sizes = dict(zip(header_sizes, row[1:]))

            elif tag == "#model_mem":
                if not header_model_mem:
                    header_model_mem = row[1:]
                else:
                    model_mem = dict(zip(header_model_mem, row[1:]))

            elif tag == "#cpu":
                if not header_cpu:
                    header_cpu = row[1:]
                else:
                    cpu_info = dict(zip(header_cpu, row[1:]))

            elif tag == "#disk":
                if not header_disk:
                    header_disk = row[1:]
                else:
                    disk_info = dict(zip(header_disk, row[1:]))

            elif tag == "#dataset":
                if not header_dataset:
                    header_dataset = row[1:]
                else:
                    dataset_info = dict(zip(header_dataset, row[1:]))

            elif tag == "#prediction":
                if not header_prediction:
                    header_prediction = row[1:]
                else:
                    prediction = dict(zip(header_prediction, row[1:]))

            elif tag == "#curve_val_f1":
                try:
                    curve_val = [float(v) for v in row[1:] if v]
                except ValueError:
                    pass

            elif tag == "#curve_train_f1":
                try:
                    curve_train = [float(v) for v in row[1:] if v]
                except ValueError:
                    pass

            elif tag == "#curve_epochs":
                try:
                    curve_epochs = [int(v) for v in row[1:] if v]
                except ValueError:
                    pass

            elif tag == "#ddp":
                if not header_ddp:
                    header_ddp = row[1:]
                else:
                    ddp_rows.append(dict(zip(header_ddp, row[1:])))

            elif tag == "#study_lr":
                if not header_study_lr:
                    header_study_lr = row[1:]
                else:
                    study["lr"] = dict(zip(header_study_lr, row[1:]))
            elif tag == "#study_lr_curve_lrs":
                study["lr_curve_lrs"] = _floats(row[1:])
            elif tag == "#study_lr_curve_losses":
                study["lr_curve_losses"] = _floats(row[1:])
            elif tag == "#study_conv":
                if not header_study_conv:
                    header_study_conv = row[1:]
                else:
                    study["conv"] = dict(zip(header_study_conv, row[1:]))
            elif tag == "#study_conv_steps":
                study["conv_steps"] = _floats(row[1:])
            elif tag == "#study_conv_losses":
                study["conv_losses"] = _floats(row[1:])
            elif tag == "#study_conv_f1s":
                study["conv_f1s"] = _floats(row[1:])
            elif tag == "#study_grad":
                if not header_study_grad:
                    header_study_grad = row[1:]
                else:
                    study["grad"] = dict(zip(header_study_grad, row[1:]))

            else:
                rows.append(row)

    # Build combined metadata
    combined: dict = {**meta, **model_mem}
    # Real dataset size (n images per split). If the CSV is old and lacks
    # #sizes, it will be absent and the comparison will use its fallback.
    for k in ("n_train", "n_val"):
        if k in sizes:
            try:
                combined[k] = int(float(sizes[k]))
            except (ValueError, TypeError):
                pass
    if "nfs_factor" in sizes:
        try:
            combined["nfs_factor"] = float(sizes["nfs_factor"])
        except (ValueError, TypeError):
            pass
    if cpu_info:
        combined["cpu"] = cpu_info
    if disk_info:
        combined["disk"] = disk_info
    if dataset_info:
        combined["dataset"] = dataset_info
    if prediction:
        combined["prediction"] = prediction
        if curve_val:
            combined["curve_val_f1"] = curve_val
        if curve_train:
            combined["curve_train_f1"] = curve_train
        if curve_epochs:
            combined["curve_epochs"] = curve_epochs
    if ddp_rows:
        combined["ddp_scenarios"] = ddp_rows
    if study:
        combined["study"] = study

    # Normalize numeric metadata fields
    float_fields = (
        "total_params_M", "flops_mflops", "total_vram_gb", "free_vram_gb",
        "weight_mb", "gradient_mb", "optimizer_mb",
        "activation_mb_per_image", "total_static_mb",
    )
    for key in float_fields:
        if key in combined:
            try:
                combined[key] = float(combined[key])
            except (ValueError, TypeError):
                pass

    # Normalize numeric prediction
    if "prediction" in combined:
        for k in ("predicted_best_f1", "predicted_best_epoch", "predicted_early_stop_epoch"):
            if k in combined["prediction"]:
                try:
                    combined["prediction"][k] = float(combined["prediction"][k])
                except (ValueError, TypeError):
                    pass

    if not rows:
        return combined, pd.DataFrame()

    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)

    # Convert numeric columns
    numeric_cols = [
        "batch_size",
        "s_per_batch", "imgs_per_s",              # legacy
        "s_per_batch_train", "imgs_per_s_train",  # v2+
        "s_per_batch_eval", "imgs_per_s_eval",    # v2+
        "peak_vram_gb", "avg_power_w",
        "optimizer_steps_per_epoch",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if col.startswith(("est_", "flops_", "energy_", "ddp_")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return combined, df


def parse_ddp_scenarios(meta: dict) -> pd.DataFrame:
    """Extracts the DDP scenarios as a DataFrame (for the web)."""
    scenarios = meta.get("ddp_scenarios", [])
    if not scenarios:
        return pd.DataFrame()
    df = pd.DataFrame(scenarios)
    numeric_cols = ["n_gpus", "batch_per_gpu", "global_batch", "workers_per_gpu",
                    "speedup", "efficiency_pct", "sync_overhead_pct",
                    "time_train_epoch_min", "time_total_h"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
