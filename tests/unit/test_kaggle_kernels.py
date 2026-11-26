"""The Kaggle kernels build valid configs from the repo ones (a broken YAML would only show
up hours later on Kaggle)."""
import importlib.util
from pathlib import Path

import pytest

from src.training.config_validator import validate_config

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load("kaggle/panda_train/run.py")


@pytest.mark.parametrize("exp", [e for batch in train.BATCHES.values() for e in batch],
                         ids=lambda e: e[0])
def test_every_experiment_config_is_valid(exp):
    name, base, overrides, strategy = train.experiment_parts(exp)
    cfg = train.build_config(REPO, name, base, overrides)
    validate_config(cfg)
    assert cfg["data"]["image_size"] == cfg["model"]["img_size"]
    gpus = 2 if strategy == "ddp" else 1                       # split strategies: one process
    # Global batch of the local runs (16); the "_b8_" pipeline run halves it on purpose.
    assert cfg["training"]["batch_size"] * gpus == (8 if "_b8_" in name else 16)
    assert cfg["checkpoint"]["dir"].endswith(name)            # one folder per experiment


def test_train_command_per_strategy(tmp_path):
    cfg = tmp_path / "c.yaml"
    cmd, env = train.train_command(cfg, "ddp")
    assert "torch.distributed.run" in cmd and "--nproc_per_node=2" in cmd and env is None
    assert "scripts/train_ddp.py" in cmd and "--precision" in cmd
    cmd, env = train.train_command(cfg, "single")
    assert "scripts/train_single_gpu.py" in cmd and "torch.distributed.run" not in cmd
    assert env == {"CUDA_VISIBLE_DEVICES": "0"}
    with pytest.raises(ValueError, match="unknown strategy"):
        train.experiment_parts(("x", "configs/train_panda.yaml", {}, "fsdp"))
    assert train.experiment_parts(("x", "b", {}))[3] == "ddp"


def test_train_command_for_the_split_strategies(tmp_path):
    cfg = tmp_path / "c.yaml"
    cmd, env = train.train_command(cfg, "model_parallel")
    assert "scripts/train_model_parallel.py" in cmd and env is None and "--monitor" not in cmd
    cmd, _ = train.train_command(cfg, "pipeline:8")
    assert "scripts/train_pipeline.py" in cmd and cmd[-2:] == ["--micro-batches", "8"]
    assert train.experiment_parts(("x", "b", {}, "pipeline:2"))[3] == "pipeline:2"


def test_overrides_are_not_shared_between_experiments():
    """single()/model() build fresh dicts: one experiment's batch must not leak into another."""
    a, b = train.single(train.model("m")), train.model("m")
    assert a["training"]["batch_size"] == 16 and "batch_size" not in b["training"]


def test_experiment_names_are_unique_across_batches():
    names = [exp[0] for batch in train.BATCHES.values() for exp in batch]
    assert len(names) == len(set(names))                      # results/<name> never overwritten
    assert train.EXPERIMENTS is train.BATCHES[train.BATCH]


def test_deep_update_merges_nested_dicts():
    base = {"data": {"a": 1, "b": 2}, "x": 1}
    assert train.deep_update(base, {"data": {"b": 3}}) == {"data": {"a": 1, "b": 3}, "x": 1}


def test_metadata_does_not_commit_the_username():
    for meta in REPO.glob("kaggle/*/kernel-metadata.json"):
        text = meta.read_text()
        assert '"{user}/' in text and "alu0" not in text


bench = _load("kaggle/benchmark_t4/run.py")


def test_benchmark_kernel_parses_the_report():
    report = """  Batch  Modo      imgs/s(train)  imgs/s(eval)     VRAM
      8  off             138.6       471.7   1.96 GB
     16  off             153.6       531.7   3.86 GB
     16  simple          150.0       500.0   3.90 GB
     16  off                 OOM
"""
    rows = bench.parse_rows(report)
    assert rows == [{"batch": 8, "train_ips": 138.6, "eval_ips": 471.7, "vram_gb": 1.96},
                    {"batch": 16, "train_ips": 153.6, "eval_ips": 531.7, "vram_gb": 3.86}]
    assert bench.parse_rows("     16  off                 OOM\n") == [
        {"batch": 16, "train_ips": None, "eval_ips": None, "vram_gb": None}]
    assert len(set(bench.MODELS)) == len(bench.MODELS) == 13


def test_allreduce_bus_bandwidth():
    spec = importlib.util.spec_from_file_location("_allreduce", REPO / "scripts" / "measure_allreduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.bus_bandwidth(10.0, 2) == pytest.approx(10.0)       # ring of 2: busbw = algbw
    assert mod.bus_bandwidth(10.0, 4) == pytest.approx(15.0)
    assert mod.bus_bandwidth(10.0, 1) == pytest.approx(10.0)


def test_benchmark_campaigns_are_well_formed():
    names = [c[0] for c in bench.CAMPAIGNS]
    assert len(set(names)) == len(names)
    for _name, img_size, precision, batches, models in bench.CAMPAIGNS:
        assert img_size in (224, 384, 512) and precision in ("amp", "fp32")
        assert batches and all(b.isdigit() for b in batches)
        assert set(models) <= set(bench.MODELS)


def test_cluster_sweep_reuses_the_t4_campaigns():
    sweep = _load("cluster/benchmark/throughput_sweep.py")
    camps = sweep.campaigns(bench)
    names = [c[0] for c in camps]
    assert names[0] == "amp_512" and names[1:] == [c[0] for c in bench.CAMPAIGNS]
    assert len(set(names)) == len(names)
    assert camps[0][1:4] == (512, "amp", ["8", "16"]) and camps[0][4] == bench.MODELS
