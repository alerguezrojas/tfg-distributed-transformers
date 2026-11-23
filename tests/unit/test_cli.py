"""Unit tests for the unified CLI command builders (src/cli.py).

Only the pure argv builders are tested — the Typer commands just assemble these
and shell out, so covering the builders covers what could break.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cli import (build_train_cmd, build_benchmark_cmd, build_eval_cmd, STRATEGIES,
                     _run_row, resolve_dataset_n)
from src.performance_model import N_FULL_TRAIN, N_SUBSET_TRAIN
from src.web.run_registry import discover_runs


def _join(cmd):
    return " ".join(cmd)


# ── train ──────────────────────────────────────────────────────────────────────

def test_train_single_plain_script():
    cmd = build_train_cmd("single", "configs/train.yaml", model="vit_tiny_patch16_224",
                          epochs=5, trace="simple")
    assert "scripts/train_single_gpu.py" in cmd
    assert "torch.distributed.run" not in _join(cmd)   # single is not under torchrun
    assert "--model" in cmd and "vit_tiny_patch16_224" in cmd
    assert "--epochs" in cmd and "5" in cmd
    assert cmd[-2:] != []  # has args


def test_train_single_precision_and_layers():
    cmd = build_train_cmd("single", "configs/train.yaml", precision="amp",
                          layers=["plot", "confusion"], fn=["energy"])
    s = _join(cmd)
    assert "--precision amp" in s
    assert "--layers plot confusion" in s
    assert "--fn energy" in s


def test_train_single_fp32_omits_precision_flag():
    cmd = build_train_cmd("single", "configs/train.yaml", precision="fp32")
    assert "--precision" not in cmd          # fp32 is the implicit default


def test_train_precision_forwarded_to_ddp_and_model_parallel():
    # single/ddp/model-parallel go through Trainer.train_epoch (autocast), so --precision
    # applies; heterogeneous runs a hand-written fp32 loop on gloo, so it must NOT.
    for strat in ("single", "ddp", "model-parallel"):
        s = _join(build_train_cmd(strat, "cfg.yaml", precision="amp"))
        assert "--precision amp" in s, strat
    s_het = _join(build_train_cmd("heterogeneous", "cfg.yaml", precision="amp"))
    assert "--precision" not in s_het


def test_train_ddp_uses_torchrun_with_nproc():
    cmd = build_train_cmd("ddp", "configs/train_demo_ddp.yaml", n_gpus=2)
    s = _join(cmd)
    assert "torch.distributed.run" in s
    assert "--nproc_per_node=2" in s
    assert "scripts/train_ddp.py" in cmd
    assert "--nnodes" not in s               # single-node by default


def test_train_ddp_multinode_adds_rendezvous():
    cmd = build_train_cmd("ddp", "c.yaml", n_gpus=1, nnodes=2, node_rank=1,
                          master_addr="verode21", master_port=29501)
    s = _join(cmd)
    assert "--nnodes=2" in s and "--node_rank=1" in s
    assert "--master_addr=verode21" in s and "--master_port=29501" in s


def test_train_model_parallel_forwards_layers_and_fn():
    # model_parallel runs through the builder, so it must honour --layers/--fn
    # (single-process, no torchrun).
    cmd = build_train_cmd("model-parallel", "configs/train_model_parallel_kaggle.yaml",
                          layers=["confusion"], fn=["energy"])
    s = _join(cmd)
    assert "scripts/train_model_parallel.py" in cmd
    assert "torch.distributed.run" not in s
    assert "--layers" in s and "confusion" in s
    assert "--fn" in s and "energy" in s


def test_train_heterogeneous_one_proc_per_node():
    cmd = build_train_cmd("heterogeneous", "configs/train_heterogeneous_ddp.yaml",
                          nnodes=2, node_rank=0, master_addr="verode21")
    s = _join(cmd)
    assert "scripts/train_heterogeneous_ddp.py" in cmd
    assert "--nproc_per_node=1" in s         # 1 process per node
    assert "--nnodes=2" in s


def test_train_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_train_cmd("quantum", "c.yaml")


def test_strategies_constant():
    assert set(STRATEGIES) == {"single", "ddp", "model_parallel", "heterogeneous"}


# ── benchmark ──────────────────────────────────────────────────────────────────

def test_benchmark_basic():
    cmd = build_benchmark_cmd(["vit_base_patch16_224"], [32, 64], 30, ["off", "simple"])
    s = _join(cmd)
    assert "scripts/benchmark.py" in cmd
    assert "--model vit_base_patch16_224" in s
    assert "--batch-sizes 32 64" in s
    assert "--trace-modes off simple" in s
    assert "--epochs 30" in s


def test_benchmark_study_and_precision():
    cmd = build_benchmark_cmd(None, [32], 15, None, precision="amp",
                                compare_precision=True, convergence_study=True,
                                study_steps=80, nfs_factor=1.3)
    s = _join(cmd)
    assert "--precision amp" in s
    assert "--compare-precision" in s
    assert "--convergence-study --study-steps 80" in s
    assert "--nfs-factor 1.3" in s


# ── eval ─────────────────────────────────────────────────────────────────────────

def test_eval_basic():
    cmd = build_eval_cmd("ckpt.pt", "configs/train.yaml", split="test",
                         output="logs/test.csv")
    s = _join(cmd)
    assert "scripts/eval.py" in cmd
    assert "--checkpoint ckpt.pt" in s
    assert "--split test" in s
    assert "--output logs/test.csv" in s


def test_eval_ensemble_repeats_checkpoint():
    cmd = build_eval_cmd(["a/best.pt", "b/best.pt", "c/best.pt"], "c.yaml",
                         save_predictions="logs/ens.npz")
    assert [cmd[i + 1] for i, t in enumerate(cmd) if t == "--checkpoint"] == [
        "a/best.pt", "b/best.pt", "c/best.pt"]
    assert "--save-predictions logs/ens.npz" in _join(cmd)


def test_eval_single_checkpoint_as_list_equals_string():
    assert build_eval_cmd(["a.pt"], "c.yaml") == build_eval_cmd("a.pt", "c.yaml")


def test_eval_forwards_data_root():
    assert "--data-root /data/x" in _join(build_eval_cmd("ckpt.pt", "c.yaml", data_root="/data/x"))
    assert "--data-root" not in _join(build_eval_cmd("ckpt.pt", "c.yaml"))


def test_eval_optional_flags():
    cmd = build_eval_cmd("ckpt.pt", "c.yaml", model="vit_tiny_patch16_224",
                         metadata="meta.parquet", max_batches=10)
    s = _join(cmd)
    assert "--model vit_tiny_patch16_224" in s
    assert "--metadata meta.parquet" in s
    assert "--max-batches 10" in s
    assert "--save-predictions" not in s


def test_eval_save_predictions():
    cmd = build_eval_cmd("ckpt.pt", "c.yaml", save_predictions="logs/pred.npz")
    assert "--save-predictions logs/pred.npz" in _join(cmd)


# ── runs summary ─────────────────────────────────────────────────────────────────

def test_run_row_reads_best_val_and_test_f1(tmp_path):
    d = tmp_path / "logs" / "local" / "single" / "vit_tiny_patch16_224"
    d.mkdir(parents=True)
    (d / "train_01062026_120000.log").write_text("x\n")
    (d / "epoch_metrics_01062026_120000.csv").write_text(
        "epoch,val_f1\n1,0.40\n2,0.55\n3,0.50\n")
    (d / "test_01062026_120000.csv").write_text(
        "class_idx,class_name,f1,precision,recall\n0,A,0.6,0.6,0.6\n\n"
        "# aggregate,loss=0.1,f1_t05=0.50,f1_opt=0.58,threshold=0.35,"
        "accuracy=0.9,precision=0.6,recall=0.6\n")
    found = discover_runs(tmp_path)
    row = _run_row(found[0])
    assert row["metric"] == "F1"
    assert row["best_val"] == 0.55         # max over epochs
    assert row["test"] == 0.58             # optimal-threshold F1 from the aggregate line
    assert row["epochs"] == 3


def test_run_row_panda_uses_qwk(tmp_path):
    # A PANDA run is judged by QWK: neither "Best Val F1" nor the F1 of the test CSV.
    d = tmp_path / "logs" / "kaggle" / "panda" / "ddp" / "vit_small_patch16_224"
    d.mkdir(parents=True)
    (d / "train_20112026_163243.log").write_text("x\n")
    (d / "epoch_metrics_20112026_163243.csv").write_text(
        "epoch,val_f1,val_qwk\n1,0.30,0.50\n2,0.35,0.71\n3,0.33,0.69\n")
    (d / "test_20112026_163243.csv").write_text(
        "class_idx,class_name,f1,precision,recall\n0,A,0.6,0.6,0.6\n\n"
        "# aggregate,loss=0.1,f1_t05=0.30,accuracy=0.5,precision=0.6,recall=0.6,auc=0.8,qwk=0.66\n")
    row = _run_row(discover_runs(tmp_path)[0])
    assert row["metric"] == "QWK"
    assert row["best_val"] == 0.71 and row["test"] == 0.66


def test_run_row_ignores_test_csv_of_another_run(tmp_path):
    d = tmp_path / "logs" / "local" / "single" / "vit_tiny_patch16_224"
    d.mkdir(parents=True)
    (d / "train_01062026_120000.log").write_text("x\n")
    (d / "train_01062026_130000.log").write_text("x\n")
    (d / "test_01062026_130000.csv").write_text(
        "class_idx,class_name,f1,precision,recall\n0,A,0.6,0.6,0.6\n\n"
        "# aggregate,loss=0.1,f1_t05=0.50,f1_opt=0.58\n")
    rows = {r.timestamp: _run_row(r) for r in discover_runs(tmp_path)}
    assert rows["01062026_120000"]["test"] is None
    assert rows["01062026_130000"]["test"] == 0.58


def test_resolve_dataset_n_full_subset_custom():
    assert resolve_dataset_n("full") == N_FULL_TRAIN
    assert resolve_dataset_n("subset") == N_SUBSET_TRAIN
    assert resolve_dataset_n("Full (237,871)") == N_FULL_TRAIN     # web label form
    assert resolve_dataset_n("subset", 1234) == 1234                # explicit override wins


def test_run_row_no_csv_is_dashes(tmp_path):
    d = tmp_path / "logs" / "kaggle" / "single" / "vit_base_patch16_224"
    d.mkdir(parents=True)
    (d / "train_02062026_120000.log").write_text("x\n")
    row = _run_row(discover_runs(tmp_path)[0])
    assert row["best_val"] is None and row["test"] is None


def test_eval_save_predictions_auto_has_no_path():
    s = _join(build_eval_cmd("ckpt.pt", "c.yaml", save_predictions_auto=True))
    assert s.endswith("--save-predictions")
    both = _join(build_eval_cmd("ckpt.pt", "c.yaml", save_predictions="p.npz",
                                save_predictions_auto=True))
    assert "--save-predictions p.npz" in both and both.count("--save-predictions") == 1


def test_train_cmd_seed_and_data_root_all_strategies():
    for strat in STRATEGIES:
        s = _join(build_train_cmd(strat, "c.yaml", seed=0, data_root="/data"))
        assert "--seed 0" in s and "--data-root /data" in s


def test_train_cmd_omits_seed_and_data_root_by_default():
    s = _join(build_train_cmd("single", "c.yaml"))
    assert "--seed" not in s and "--data-root" not in s


def test_train_dry_run_passes_seed_and_data_root():
    from typer.testing import CliRunner
    from src.cli import app
    r = CliRunner().invoke(app, ["train", "--config", "c.yaml", "--seed", "42",
                                 "--data-root", "/d", "--dry-run"])
    assert r.exit_code == 0
    out = r.output.replace("\n", " ")
    assert "--seed 42" in out and "--data-root /d" in out


def test_benchmark_cmd_passes_split_sizes():
    s = _join(build_benchmark_cmd(None, None, 15, None, train_size=7433, val_size=1593))
    assert "--train-size 7433" in s and "--val-size 1593" in s
    assert "--train-size" not in _join(build_benchmark_cmd(None, None, 15, None))


# ── --machine / --set (perfiles de máquina) ─────────────────────────────────────

def test_train_forwards_machine_and_set():
    cmd = build_train_cmd("single", "c.yaml", machine="verode",
                          overrides=["training.lr=1e-4", "data.num_workers=2"])
    assert cmd[cmd.index("--machine") + 1] == "verode"
    assert [cmd[j + 1] for j, a in enumerate(cmd) if a == "--set"] == [
        "training.lr=1e-4", "data.num_workers=2"]


def test_train_without_machine_adds_nothing():
    cmd = build_train_cmd("ddp", "c.yaml")
    assert "--machine" not in cmd and "--set" not in cmd


def test_eval_benchmark_predict_forward_machine():
    from src.cli import build_predict_cmd
    assert build_eval_cmd("a.pt", "c.yaml", machine="local")[-2:] == ["--machine", "local"]
    assert build_benchmark_cmd(None, None, 1, None, overrides=["a.b=1"])[-2:] == ["--set", "a.b=1"]
    cmd = build_predict_cmd("a.pt", "c.yaml", ["img.png"], machine="kaggle", overrides=["x=1"])
    assert "--machine" in cmd and cmd[-1].endswith("img.png")


def test_cli_train_dry_run_shows_machine_and_set():
    from typer.testing import CliRunner
    from src.cli import app
    res = CliRunner().invoke(app, ["train", "--machine", "local", "--set", "training.lr=1e-4",
                                   "--dry-run"])
    assert res.exit_code == 0
    assert "--machine" in res.output and "training.lr=1e-4" in res.output
