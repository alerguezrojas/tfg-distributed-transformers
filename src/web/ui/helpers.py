"""Cached data loaders and general helpers for the dashboard."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from src.web.batch_parser import parse_batch_csv
from src.web.dataset_stats import (
    CLASS_NAMES, class_distribution_from_parquet, find_example_patches, load_rgb_image,
)
from src.web.feasibility_parser import parse_feasibility_csv
from src.web.log_parser import parse_log
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_registry import RunInfo, discover_feasibility_csvs, discover_runs

ROOT = Path(__file__).resolve().parents[3]

# ── Cached loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_df(log_path: str, epoch_csv: str | None) -> pd.DataFrame:
    if epoch_csv and Path(epoch_csv).exists():
        df = pd.read_csv(epoch_csv)
        if not df.empty:
            if "epoch_time_s" in df.columns:
                df = df.rename(columns={"epoch_time_s": "epoch_time"})
            # Energy/power and timings only live in the log (they are not
            # written to the epoch_metrics CSV). If the log exists, we merge
            # them by epoch so the energy panel also shows up with the CSV.
            _energy_cols = ["energy_train_j", "energy_eval_j", "energy_eval_wh",
                            "power_train_w", "power_eval_w", "time_train_s", "time_eval_s"]
            missing = [c for c in _energy_cols if c not in df.columns
                       or not df[c].notna().any()]
            if missing and log_path and Path(log_path).exists():
                log_df = parse_log(Path(log_path))
                merge_cols = [c for c in missing if c in log_df.columns]
                if merge_cols and "epoch" in log_df.columns:
                    df = df.merge(
                        log_df[["epoch", *merge_cols]], on="epoch", how="left"
                    )
            return df
    return parse_log(Path(log_path))


@st.cache_data(ttl=30)
def _load_batch(csv_path: str) -> pd.DataFrame:
    return parse_batch_csv(Path(csv_path))


@st.cache_data(ttl=30)
def _load_perclass(csv_path: str) -> pd.DataFrame:
    return parse_perclass_csv(Path(csv_path))


@st.cache_data(ttl=60)
def _get_runs() -> list[RunInfo]:
    return discover_runs(ROOT)


@st.cache_data(ttl=60)
def _get_feasibility_csvs() -> list[Path]:
    return discover_feasibility_csvs(ROOT)


@st.cache_data(ttl=60)
def _feas_label(path_str: str) -> str:
    """Readable label for a feasibility CSV: 'env · model · DD/MM HH:MM'
    instead of the raw date-based filename."""
    import re
    p = Path(path_str)
    env = p.parent.parent.name if p.parent.parent else "?"
    try:
        m, _ = parse_feasibility_csv(p)
        model = str(m.get("model_name", "?")).replace("_patch16_224", "")
    except Exception:
        model = "?"
    mt = re.search(r"(\d{2})(\d{2})\d{4}_(\d{2})(\d{2})", p.name)
    when = f"{mt.group(1)}/{mt.group(2)} {mt.group(3)}:{mt.group(4)}" if mt else p.stem
    return f"{env} · {model} · {when}"


@st.cache_data(ttl=30)
def _run_config(log_path_str: str) -> dict:
    """Extracts the 'Configuración: k=v | k=v | ...' line from the log → dict.
    Returns {} if the run predates this version (it does not record it).
    Note: the log key stays 'Configuración:' to match existing/backfilled logs."""
    try:
        for line in Path(log_path_str).read_text(errors="replace").splitlines():
            i = line.find("Configuración:")
            if i < 0:
                continue
            out = {}
            for part in line[i + len("Configuración:"):].split("|"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k.strip()] = v.strip()
            return out
    except Exception:
        pass
    return {}


# ── Cached dataset loaders ──────────────────────────────────────────────────────


@st.cache_data(ttl=600)
def _load_class_distribution(parquet_str: str) -> pd.DataFrame | None:
    """Cached class distribution (iterates ~237K rows, slow)."""
    return class_distribution_from_parquet(Path(parquet_str))


@st.cache_data(ttl=600)
def _load_example_images(parquet_str: str, root_str: str, class_name: str, n: int = 4):
    """Loads n example RGB images for a class, cached."""
    patches = find_example_patches(Path(parquet_str), class_name, n=n)
    images = []
    for pid in patches:
        img = load_rgb_image(Path(root_str), pid)
        if img is not None:
            images.append((pid, img))
    return images


@st.cache_data(ttl=900)
def _class_gallery(parquet_str: str, root_str: str):
    """One example RGB image per class + its train statistics, in a SINGLE pass
    over the parquet (find_example_patches re-reads it per class — too slow for
    19 classes). Returns [(class_name, count, pct_of_train_patches, image)]."""
    try:
        df = pd.read_parquet(parquet_str, columns=["patch_id", "labels", "split"])
    except Exception:
        return []
    df = df[df["split"] == "train"]
    n_train = max(len(df), 1)
    first_pid: dict[str, str] = {}
    counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for pid, arr in zip(df["patch_id"], df["labels"]):
        if arr is None:
            continue
        for c in arr:
            if c in counts:
                counts[c] += 1
                first_pid.setdefault(c, pid)
    out = []
    for c in CLASS_NAMES:
        pid = first_pid.get(c)
        if not pid:
            continue
        img = load_rgb_image(Path(root_str), pid)
        if img is not None:
            out.append((c, counts[c], counts[c] / n_train * 100, img))
    return out


# ── General helpers ─────────────────────────────────────────────────────────────


def _safe_max(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.max()) if not valid.empty else float("nan")


def _safe_idxmax(series: pd.Series):
    valid = series.dropna()
    return valid.idxmax() if not valid.empty else None


def _safe_val_at_best(df: pd.DataFrame, metric_col: str, target_col: str):
    if metric_col not in df.columns or target_col not in df.columns:
        return None
    idx = _safe_idxmax(df[metric_col])
    if idx is None:
        return None
    v = df.loc[idx, target_col]
    return None if pd.isna(v) else v


def _throughput_col(df: pd.DataFrame) -> str | None:
    for col in ("imgs_per_s_train", "imgs_per_s"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _dur_str(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def _get_configs() -> list[str]:
    cfg_dir = ROOT / "configs"
    if not cfg_dir.exists():
        return []
    return sorted(p.name for p in cfg_dir.glob("*.yaml"))


def _detect_anomalies(log_path: Path) -> list[str]:
    keywords = ["EXPLODE", "VANISH", "DEAD", "OOM", "explosivo", "evanescente", "muertas"]
    hits: list[str] = []
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            if any(kw in line for kw in keywords):
                hits.append(line.strip())
    except Exception:
        pass
    return hits


def _read_log_tail(log_path: Path, n: int = 40) -> str:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _parse_log_progress(log_path: Path) -> dict:
    import re
    result = {"epoch": 0, "epochs": 0, "last_val_f1": None, "last_val_loss": None}
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        for line in reversed(lines):
            if "Epoch" in line and "/" in line:
                mm = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                if mm:
                    result["epoch"] = int(mm.group(1))
                    result["epochs"] = int(mm.group(2))
                    break
        for line in reversed(lines):
            if "val_f1" in line or "val=0." in line:
                mm = re.search(r"val_f1[=\s]+([\d.]+)", line)
                if mm:
                    result["last_val_f1"] = float(mm.group(1))
                mm2 = re.search(r"val_loss[=\s]+([\d.]+)", line)
                if mm2:
                    result["last_val_loss"] = float(mm2.group(1))
                break
    except Exception:
        pass
    return result


def _gpu_usage() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) < 5:
            return None
        return {
            "name": parts[0], "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]), "util_pct": int(parts[3]),
            "temp_c": int(parts[4]),
        }
    except Exception:
        return None


def _launch_process(cmd: str, placeholder) -> int:
    output_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(ROOT),
        )
        for raw in proc.stdout:  # type: ignore[union-attr]
            output_lines.append(raw.rstrip())
            placeholder.code("\n".join(output_lines[-120:]), language="text")
        proc.wait()
        return proc.returncode
    except Exception as exc:
        placeholder.error(str(exc))
        return -1


def _color_f1_cell(v: float) -> str:
    if v >= 0.6:
        return "background-color: #d1fae5; color: #065f46"
    if v >= 0.3:
        return "background-color: #fef3c7; color: #92400e"
    return "background-color: #fee2e2; color: #991b1b"

