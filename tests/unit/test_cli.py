"""Unit tests for the unified CLI command builders (src/cli.py).

Only the pure argv builders are tested — the Typer commands just assemble these
and shell out, so covering the builders covers what could break.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cli import build_train_cmd, build_feasibility_cmd, build_eval_cmd, STRATEGIES, _run_row
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


def test_train_model_parallel_plain_no_layers():
    cmd = build_train_cmd("model-parallel", "configs/train_model_parallel_kaggle.yaml",
                          layers=["plot"])
    s = _join(cmd)
    assert "scripts/train_model_parallel.py" in cmd
    assert "torch.distributed.run" not in s
    assert "--layers" not in s               # model_parallel does not accept --layers


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
    assert set(STRATEGIES) == {"single", "ddp", "model-parallel", "heterogeneous"}


# ── feasibility ──────────────────────────────────────────────────────────────────

def test_feasibility_basic():
    cmd = build_feasibility_cmd(["vit_base_patch16_224"], [32, 64], 30, ["off", "simple"])
    s = _join(cmd)
    assert "scripts/check_feasibility.py" in cmd
    assert "--model vit_base_patch16_224" in s
    assert "--batch-sizes 32 64" in s
    assert "--trace-modes off simple" in s
    assert "--epochs 30" in s


def test_feasibility_study_and_precision():
    cmd = build_feasibility_cmd(None, [32], 15, None, precision="amp",
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


def test_eval_optional_flags():
    cmd = build_eval_cmd("ckpt.pt", "c.yaml", model="vit_tiny_patch16_224",
                         metadata="meta.parquet", max_batches=10)
    s = _join(cmd)
    assert "--model vit_tiny_patch16_224" in s
    assert "--metadata meta.parquet" in s
    assert "--max-batches 10" in s


# ── runs summary ─────────────────────────────────────────────────────────────────

def test_run_row_reads_best_val_and_test_f1(tmp_path):
    d = tmp_path / "logs" / "local" / "single" / "vit_tiny_patch16_224"
    d.mkdir(parents=True)
    (d / "train_01062026_120000.log").write_text("x\n")
    (d / "epoch_metrics_01062026_120000.csv").write_text(
        "epoch,val_f1\n1,0.40\n2,0.55\n3,0.50\n")
    (d / "test_demo.csv").write_text(
        "class_idx,class_name,f1,precision,recall\n0,A,0.6,0.6,0.6\n\n"
        "# aggregate,loss=0.1,f1_t05=0.50,f1_opt=0.58,threshold=0.35,"
        "accuracy=0.9,precision=0.6,recall=0.6\n")
    found = discover_runs(tmp_path)
    row = _run_row(found[0])
    assert row["best_val_f1"] == 0.55      # max over epochs
    assert row["test_f1"] == 0.58          # optimal-threshold F1 from the aggregate line
    assert row["epochs"] == 3


def test_run_row_no_csv_is_dashes(tmp_path):
    d = tmp_path / "logs" / "kaggle" / "single" / "vit_base_patch16_224"
    d.mkdir(parents=True)
    (d / "train_02062026_120000.log").write_text("x\n")
    row = _run_row(discover_runs(tmp_path)[0])
    assert row["best_val_f1"] is None and row["test_f1"] is None
