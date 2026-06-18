"""Tests for the held-out test-set wiring: eval CSV parser + run discovery."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.web.eval_parser import parse_eval_csv
from src.web.run_registry import discover_runs

_EVAL_CSV = """class_idx,class_name,f1,precision,recall
0,Urban fabric,0.4324,0.2857,0.8889
6,Land principally occupied by agriculture,0.0,0.0,0.0
18,Marine waters,0.878,0.8571,0.9

# aggregate,loss=0.265015,f1_t05=0.2766,f1_opt=0.3858,threshold=0.25,accuracy=0.8926,precision=0.4305,recall=0.2356
"""


def test_parse_eval_csv_perclass_and_aggregate(tmp_path):
    p = tmp_path / "test_bce.csv"
    p.write_text(_EVAL_CSV)
    pcdf, agg = parse_eval_csv(p)
    assert len(pcdf) == 3
    assert set(pcdf.columns) >= {"class_idx", "class_name", "f1", "precision", "recall"}
    assert agg["f1_opt"] == 0.3858
    assert agg["threshold"] == 0.25
    assert agg["accuracy"] == 0.8926
    # the F1=0 class is preserved (the rare-class finding)
    assert (pcdf["f1"] == 0.0).sum() == 1


def test_parse_eval_csv_missing_file(tmp_path):
    pcdf, agg = parse_eval_csv(tmp_path / "nope.csv")
    assert pcdf.empty and agg == {}


def test_discover_associates_test_csv_by_folder(tmp_path):
    # A run log + a test CSV in the same model folder are linked.
    d = tmp_path / "logs" / "verode" / "single" / "vit_base_patch16_224"
    d.mkdir(parents=True)
    (d / "train_27052026_210223.log").write_text("Configuración: modelo\n")
    (d / "test_bce.csv").write_text(_EVAL_CSV)
    (d / "test_focal.csv").write_text(_EVAL_CSV)

    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    names = sorted(p.name for p in runs[0].test_csv_paths)
    assert names == ["test_bce.csv", "test_focal.csv"]


def test_discover_no_test_csv_is_empty_list(tmp_path):
    d = tmp_path / "logs" / "local" / "single" / "vit_tiny_patch16_224"
    d.mkdir(parents=True)
    (d / "train_01062026_120000.log").write_text("x\n")
    runs = discover_runs(tmp_path)
    assert runs[0].test_csv_paths == []
