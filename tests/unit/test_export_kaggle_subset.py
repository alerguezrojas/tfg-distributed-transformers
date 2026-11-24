"""scripts/export_kaggle_subset.py takes its paths from the machine profile, not from arguments."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.machine import MachineProfile  # noqa: E402

_spec = importlib.util.spec_from_file_location("export_kaggle_subset",
                                               ROOT / "scripts/export_kaggle_subset.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)


def test_paths_from_a_profile_with_root_and_metadata_dir():
    prof = MachineProfile(name="x", data_roots={"bigearthnet": {
        "root": "/data/ben/BigEarthNet-S2", "metadata_dir": "/data/ben"}})
    root, metadata = export.bigearthnet_paths(prof)
    assert root == Path("/data/ben/BigEarthNet-S2")
    assert metadata == Path("/data/ben/metadata.parquet")


def test_profile_without_bigearthnet_is_a_clear_error():
    with pytest.raises(SystemExit, match="data_roots.bigearthnet"):
        export.bigearthnet_paths(MachineProfile(name="x", data_roots={}))
