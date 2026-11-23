"""Unit tests for scripts/predict.py — classify loose images with a trained checkpoint.

A tiny timm model (resnet18, random weights) is saved with the checkpoint format of
``Trainer._checkpoint_state``; the images are synthetic PNGs. All on CPU, no dataset.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("predict_script", ROOT / "scripts" / "predict.py")
predict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict)

from src.data.registry import dataset_spec_from_config  # noqa: E402
from src.training.tasks import MultiLabelTask  # noqa: E402
from eval import build_eval_model  # noqa: E402

N_CLASSES = 6        # PANDA: ISUP 0-5


def _cfg(name="panda", **data):
    return {"data": {"name": name, "root": "unused", "image_size": 32, **data},
            "model": {"name": "resnet18"}, "training": {}}


def _save_checkpoint(path, cfg, seed, metrics=None):
    spec = dataset_spec_from_config(cfg)
    torch.manual_seed(seed)
    model = build_eval_model(cfg, "resnet18", spec)
    torch.save({"epoch": 3, "model_state_dict": model.state_dict(), "metrics": metrics or {},
                "model_name": "resnet18"}, path)


def _images(folder, n=3, size=40):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = folder / f"img_{i}.png"
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


@pytest.fixture
def setup(tmp_path):
    cfg = _cfg()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    ck1, ck2 = tmp_path / "a.pt", tmp_path / "b.pt"
    _save_checkpoint(ck1, cfg, 1)
    _save_checkpoint(ck2, cfg, 2)
    return cfg, cfg_path, ck1, ck2, _images(tmp_path / "imgs")


def _run(cfg_path, cks, inputs, *extra):
    args = ["--config", str(cfg_path), *[a for c in cks for a in ("--checkpoint", str(c))],
            *map(str, extra), *map(str, inputs)]
    predict.main(args)


# ── Inputs ─────────────────────────────────────────────────────────────────────

def test_collect_inputs_folder_is_sorted_recursive_and_deduplicated(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("b.png", "a.jpg", "sub/c.TIF", "notes.txt"):
        Image.new("RGB", (4, 4)).save(tmp_path / name, format="PNG")
    files = predict.collect_inputs([tmp_path, tmp_path / "b.png"])
    assert [f.name for f in files] == ["a.jpg", "b.png", "c.TIF"]


def test_collect_inputs_errors_are_clear(tmp_path):
    with pytest.raises(ValueError, match="no contiene imágenes"):
        predict.collect_inputs([tmp_path])
    txt = tmp_path / "x.txt"
    txt.write_text("hola")
    with pytest.raises(ValueError, match="no es un fichero de imagen"):
        predict.collect_inputs([txt])
    with pytest.raises(ValueError, match="no existe"):
        predict.collect_inputs([tmp_path / "missing.png"])


def test_walk_skips_links_that_leave_the_folder(tmp_path):
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    Image.new("RGB", (4, 4)).save(inside / "ok.png")
    Image.new("RGB", (4, 4)).save(outside / "secret.png")
    (inside / "link.png").symlink_to(outside / "secret.png")
    (inside / "dirlink").symlink_to(outside, target_is_directory=True)
    assert [f.name for f in predict.collect_inputs([inside])] == ["ok.png"]


# ── Output table ───────────────────────────────────────────────────────────────

def test_ordinal_table_has_expected_grade_and_all_probabilities():
    spec = dataset_spec_from_config(_cfg())
    probs = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0, 0.0], [1.0, 0, 0, 0, 0, 0]])
    t = predict.build_table(["a", "b"], probs, spec.classes, spec.make_task(), 0.5)
    assert list(t["predicted"]) == [spec.classes[2], spec.classes[0]]
    assert list(t["expected_grade"]) == [2.5, 0.0]
    assert [c for c in t if c.startswith("p_")] == [f"p_{c}" for c in spec.classes]


def test_multilabel_lists_classes_above_the_threshold():
    probs = torch.tensor([[0.9, 0.2, 0.6], [0.1, 0.1, 0.1]])
    t = predict.build_table(["a", "b"], probs, ("x", "y", "z"), MultiLabelTask(), 0.5)
    assert list(t["predicted"]) == ["x;z", "-"]
    assert "expected_grade" not in t


def test_threshold_comes_from_the_checkpoint_or_defaults():
    task = MultiLabelTask()
    infos = [{"epoch": 1, "metrics": {"optimal_threshold": 0.35}}]
    assert predict.decision_threshold(task, infos) == (0.35, True)
    assert predict.decision_threshold(task, [{"epoch": 1, "metrics": {}}]) == (0.5, False)


# ── End to end ─────────────────────────────────────────────────────────────────

def test_predict_prints_table_and_writes_csv(setup, tmp_path, capsys):
    _, cfg_path, ck1, _, imgs = setup
    out = tmp_path / "o" / "preds.csv"
    _run(cfg_path, [ck1], [tmp_path / "imgs"], "--out", out, "--batch-size", "2")
    shown = capsys.readouterr().out
    assert "expected_grade" in shown and "img_0.png" in shown and "resnet18" in shown
    df = pd.read_csv(out)
    assert len(df) == 3 and list(df["file"]) == [str(p) for p in imgs]
    probs = df[[c for c in df if c.startswith("p_")]].to_numpy()
    assert probs.shape == (3, N_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
    assert np.allclose(df["prob"], probs.max(axis=1), atol=1e-3)
    assert np.allclose(df["expected_grade"], (probs * np.arange(N_CLASSES)).sum(axis=1), atol=1e-2)


def test_ensemble_is_the_mean_of_the_members(setup, tmp_path):
    _, cfg_path, ck1, ck2, _ = setup
    outs = {}
    for name, cks in (("a", [ck1]), ("b", [ck2]), ("ab", [ck1, ck2])):
        outs[name] = tmp_path / f"{name}.csv"
        _run(cfg_path, cks, [tmp_path / "imgs"], "--out", outs[name])
    cols = None
    p = {}
    for name, path in outs.items():
        df = pd.read_csv(path)
        cols = [c for c in df if c.startswith("p_")]
        p[name] = df[cols].to_numpy()
    assert not np.allclose(p["a"], p["b"])
    assert np.allclose(p["ab"], (p["a"] + p["b"]) / 2, atol=2e-5)


def test_single_image_matches_the_same_image_in_a_batch(setup, tmp_path):
    _, cfg_path, ck1, _, imgs = setup
    _run(cfg_path, [ck1], [imgs[1]], "--out", tmp_path / "one.csv")
    _run(cfg_path, [ck1], imgs, "--out", tmp_path / "all.csv")
    one, allp = pd.read_csv(tmp_path / "one.csv"), pd.read_csv(tmp_path / "all.csv")
    cols = [c for c in one if c.startswith("p_")]
    assert np.allclose(one[cols].to_numpy()[0], allp[cols].to_numpy()[1], atol=1e-4)


def test_unreadable_image_fails_clearly(setup, tmp_path):
    _, cfg_path, ck1, _, _ = setup
    fake = tmp_path / "fake.png"
    fake.write_text("esto no es un PNG")
    with pytest.raises(SystemExit, match="fake.png"):
        _run(cfg_path, [ck1], [fake])


def test_empty_folder_and_non_image_fail_before_loading(setup, tmp_path):
    _, cfg_path, ck1, _, _ = setup
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no contiene imágenes"):
        _run(cfg_path, [ck1], [empty])
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(SystemExit, match="no es un fichero de imagen"):
        _run(cfg_path, [ck1], [txt])


def test_out_must_be_csv(setup, tmp_path):
    _, cfg_path, ck1, _, imgs = setup
    with pytest.raises(SystemExit, match=r"\.csv"):
        _run(cfg_path, [ck1], imgs, "--out", tmp_path / "x.npz")


def test_dataset_without_image_reader_is_rejected(tmp_path):
    cfg = {"data": {"name": "bigearthnet", "root": "r", "metadata": "m"},
           "model": {"name": "resnet18"}}
    cfg_path = tmp_path / "ben.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    img = _images(tmp_path / "i", n=1)[0]
    with pytest.raises(SystemExit, match="no se puede leer desde imágenes sueltas"):
        _run(cfg_path, [tmp_path / "unused.pt"], [img])


def test_medmnist_reader_gives_grayscale_or_rgb_arrays(tmp_path):
    from src.data import medmnist
    p = tmp_path / "g.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p)
    assert medmnist.read_image(p, 1).shape == (8, 8)
    assert medmnist.read_image(p, 3).shape == (8, 8, 3)


# ── CLI ────────────────────────────────────────────────────────────────────────

def test_cli_predict_dry_run_builds_the_command():
    from typer.testing import CliRunner
    from src.cli import app, build_predict_cmd
    r = CliRunner().invoke(app, ["predict", "a.png", "dir/", "--checkpoint", "x.pt",
                                 "--checkpoint", "y.pt", "--config", "c.yaml", "--out", "p.csv",
                                 "--batch-size", "4", "--device", "cpu", "--dry-run"])
    assert r.exit_code == 0
    assert "not executed" in r.output
    cmd = build_predict_cmd(["x.pt", "y.pt"], "c.yaml", ["a.png", "dir/"], out="p.csv",
                            batch_size=4, device="cpu")
    assert cmd[1] == "scripts/predict.py" and cmd[-2:] == ["a.png", "dir/"]
    assert cmd[2:6] == ["--checkpoint", "x.pt", "--checkpoint", "y.pt"]
    assert "--out p.csv --batch-size 4 --device cpu" in " ".join(cmd)
    assert build_predict_cmd("x.pt", "c.yaml", ["a.png"])[-1] == "a.png"
