"""Import training artifacts produced on another machine (Kaggle, Verode…).

The Kaggle/cluster workflow trains elsewhere and downloads a zip of ``logs/``.
This module copies the artifacts into the repo's ``logs/`` tree so the dashboard
discovers them like any local run — no manual unzipping into the right folder.

Pure functions (no Streamlit), so they are unit-testable without a GPU or a UI.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

# Files we accept; everything else in the archive is ignored.
_ARTIFACT_RE = re.compile(
    r"^(train_.*\.log"
    r"|epoch_metrics_.*\.csv"
    r"|batch_metrics_.*\.csv"
    r"|perclass_metrics_.*\.csv"
    r"|confusion_matrix_.*\.csv"
    r"|feasibility_.*\.(csv|log))$"
)


def _dest_relpath(member_path: str) -> Path | None:
    """Maps an archive member path to its destination relative to ``logs/``.

    Accepts both a zip made FROM ``logs/`` (paths contain a ``logs/`` segment)
    and one made from its CONTENTS (e.g. ``kaggle/single/<model>/train_*.log``,
    treated as already relative to ``logs/``). Returns None for non-artifacts or
    path-traversal attempts.
    """
    base = member_path.split("/")[-1]
    if not _ARTIFACT_RE.match(base):
        return None
    parts = [p for p in member_path.split("/") if p not in ("", ".")]
    if ".." in parts:                       # refuse path traversal
        return None
    if "logs" in parts:                     # slice from the logs/ segment
        parts = parts[parts.index("logs") + 1:]
    rel = Path(*parts) if parts else Path(base)
    return rel


def import_run_archive(file_bytes: bytes, logs_root: Path) -> list[str]:
    """Extracts the known artifacts from a zip into ``logs_root``.

    Returns the list of imported paths relative to ``logs/`` (so the caller can
    report what landed and which runs/feasibility reports it produced).
    """
    imported: list[str] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            rel = _dest_relpath(name)
            if rel is None:
                continue
            dest = (logs_root / rel).resolve()
            if not str(dest).startswith(str(logs_root.resolve())):
                continue                    # belt-and-suspenders against traversal
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src:
                dest.write_bytes(src.read())
            imported.append(str(rel))
    return imported


def import_run_folder(folder: Path, logs_root: Path) -> list[str]:
    """Copies known artifacts found under ``folder`` into ``logs_root``.

    Mirrors the ``{env}/{mode}/{model}/`` tail when the folder already follows
    the project layout (a ``logs/`` segment in the path); otherwise the file is
    placed directly under ``logs/``.
    """
    folder = Path(folder)
    imported: list[str] = []
    if not folder.is_dir():
        return imported
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or not _ARTIFACT_RE.match(path.name):
            continue
        rel = _dest_relpath(str(path))
        if rel is None:
            continue
        dest = (logs_root / rel).resolve()
        if not str(dest).startswith(str(logs_root.resolve())):
            continue
        if dest == path.resolve():          # importing from inside logs/ itself
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        imported.append(str(rel))
    return imported


def summarize_import(rel_paths: list[str]) -> dict[str, int]:
    """Counts imported artifacts by kind, for a friendly post-import message."""
    summary = {"runs": 0, "feasibility": 0, "metric_csvs": 0, "total": len(rel_paths)}
    for p in rel_paths:
        base = Path(p).name
        if base.startswith("train_") and base.endswith(".log"):
            summary["runs"] += 1
        elif base.startswith("feasibility_"):
            summary["feasibility"] += 1
        elif base.endswith(".csv"):
            summary["metric_csvs"] += 1
    return summary
