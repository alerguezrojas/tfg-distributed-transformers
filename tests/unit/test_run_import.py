"""Unit tests for src/web/run_import.py — importing remote training artifacts."""
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.web.run_import import (
    _dest_relpath, import_run_archive, import_run_folder, summarize_import,
)


def _make_zip(names_to_content: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in names_to_content.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_dest_relpath_slices_from_logs_segment():
    assert _dest_relpath("logs/kaggle/single/vit_base/train_1.log") == \
        Path("kaggle/single/vit_base/train_1.log")
    # Zip made from logs/ CONTENTS (no logs/ prefix) → already relative.
    assert _dest_relpath("kaggle/ddp/vit_base/epoch_metrics_2.csv") == \
        Path("kaggle/ddp/vit_base/epoch_metrics_2.csv")


def test_dest_relpath_rejects_non_artifacts_and_traversal():
    assert _dest_relpath("logs/kaggle/single/README.md") is None
    assert _dest_relpath("logs/notes.txt") is None
    assert _dest_relpath("../../etc/train_evil.log") is None


def test_import_archive_writes_artifacts(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    zip_bytes = _make_zip({
        "logs/kaggle/single/vit_base_patch16_224/train_10062026_173904.log": b"Configuracion: x\n",
        "logs/kaggle/single/vit_base_patch16_224/epoch_metrics_10062026_173904.csv": b"epoch\n1\n",
        "logs/kaggle/feasibility/feasibility_10062026_172939.csv": b"#meta\n",
        "logs/kaggle/single/vit_base_patch16_224/notes.txt": b"ignored",   # not an artifact
        "logs/": b"",                                                       # dir entry
    })
    rel = import_run_archive(zip_bytes, logs)
    assert "kaggle/single/vit_base_patch16_224/train_10062026_173904.log" in rel
    assert (logs / "kaggle/single/vit_base_patch16_224/epoch_metrics_10062026_173904.csv").exists()
    assert (logs / "kaggle/feasibility/feasibility_10062026_172939.csv").exists()
    assert not (logs / "kaggle/single/vit_base_patch16_224/notes.txt").exists()
    assert len(rel) == 3


def test_import_archive_handles_contents_zip(tmp_path):
    """A zip made from inside logs/ (no logs/ prefix) still lands correctly."""
    logs = tmp_path / "logs"
    logs.mkdir()
    zip_bytes = _make_zip({
        "kaggle/ddp/vit_base_patch16_224/train_deep_10062026_205332.log": b"x\n",
    })
    rel = import_run_archive(zip_bytes, logs)
    assert (logs / "kaggle/ddp/vit_base_patch16_224/train_deep_10062026_205332.log").exists()
    assert rel == ["kaggle/ddp/vit_base_patch16_224/train_deep_10062026_205332.log"]


def test_import_folder(tmp_path):
    src = tmp_path / "downloaded" / "logs" / "verode" / "single" / "vit_tiny"
    src.mkdir(parents=True)
    (src / "train_01012026_000000.log").write_text("x")
    (src / "perclass_metrics_01012026_000000.csv").write_text("class\n")
    (src / "ignore.json").write_text("{}")
    logs = tmp_path / "repo" / "logs"
    logs.mkdir(parents=True)
    rel = import_run_folder(tmp_path / "downloaded", logs)
    assert (logs / "verode/single/vit_tiny/train_01012026_000000.log").exists()
    assert len(rel) == 2


def test_summarize_import():
    s = summarize_import([
        "kaggle/single/m/train_1.log",
        "kaggle/single/m/train_deep_2.log",
        "kaggle/single/m/epoch_metrics_1.csv",
        "kaggle/feasibility/feasibility_3.csv",
    ])
    assert s["runs"] == 2
    assert s["feasibility"] == 1
    assert s["metric_csvs"] == 1
    assert s["total"] == 4
