"""Registro de estrategias (StrategySpec), opciones comunes del lanzador y líneas de configuración."""
import pytest

from src.cli import STRATEGIES, build_train_cmd
from src.training.launch import build_parser, check_options
from src.training.strategies import (
    LaunchContext, StrategySpec, available_strategies, get_strategy, register_strategy)

CANONICAL = ["single", "ddp", "heterogeneous", "model_parallel"]
COMMON_OPTIONS = ["--config", "--epochs", "--model", "--precision", "--batch-size", "--monitor",
                  "--trace", "--inspect", "--layers", "--fn", "--metrics", "--batch-log-every",
                  "--resume", "--data-root", "--seed", "--machine", "--set"]


# ── Registro ──────────────────────────────────────────────────────────────────

def test_registry_lists_the_canonical_names():
    assert available_strategies() == CANONICAL
    assert list(STRATEGIES) == CANONICAL            # the CLI reads them from the registry


@pytest.mark.parametrize("alias,name", [
    ("model-parallel", "model_parallel"), ("Model_Parallel", "model_parallel"), ("mp", "model_parallel"),
    ("heterogeneous_ddp", "heterogeneous"), ("hetero", "heterogeneous"),
    ("single_gpu", "single"), ("ddp", "ddp")])
def test_aliases_resolve_to_the_canonical_spec(alias, name):
    assert get_strategy(alias).name == name


def test_unknown_strategy_lists_the_available_ones():
    with pytest.raises(ValueError, match="available"):
        get_strategy("quantum")


def _spec(name="demo", **kw):
    hook = lambda ctx: None  # noqa: E731
    return StrategySpec(name=name, description="d", script="scripts/x.py", default_config="c.yaml",
                        init_env=hook, make_loaders=hook, make_builder=hook,
                        config_line=hook, **kw)


def test_register_rejects_duplicates_and_taken_aliases():
    register_strategy(spec := _spec())
    try:
        assert register_strategy(spec) is spec           # idempotent for the same object
        with pytest.raises(ValueError, match="already registered"):
            register_strategy(_spec())
        with pytest.raises(ValueError, match="alias"):
            register_strategy(_spec("other", aliases=("mp",)))
    finally:
        from src.training import strategies
        strategies._REGISTRY.pop("demo", None)


def test_spec_validates_its_name_and_launcher():
    with pytest.raises(ValueError, match="lowercase"):
        _spec("Model-Parallel")
    with pytest.raises(ValueError, match="launcher"):
        _spec(launcher="mpirun")
    with pytest.raises(ValueError, match="distributed"):
        _spec(launcher="torchrun")


def test_launch_modes():
    assert [get_strategy(n).torchrun for n in CANONICAL] == [False, True, True, False]
    assert get_strategy("heterogeneous").nproc == 1 and get_strategy("ddp").nproc is None


# ── Opciones comunes ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", CANONICAL)
def test_every_strategy_accepts_every_common_option(name):
    parser = build_parser(get_strategy(name))
    flags = {opt for action in parser._actions for opt in action.option_strings}
    assert set(COMMON_OPTIONS) <= flags


def test_model_parallel_has_its_own_options_and_nobody_else():
    own = {"--split-block", "--devices"}
    for name in CANONICAL:
        flags = {o for a in build_parser(get_strategy(name))._actions for o in a.option_strings}
        assert own <= flags if name == "model_parallel" else not (own & flags)


def test_default_config_is_per_strategy():
    defaults = {n: build_parser(get_strategy(n)).parse_args([]).config for n in CANONICAL}
    assert defaults["single"] == "configs/train.yaml"
    assert defaults["ddp"] == "configs/train_cluster.yaml"
    assert defaults["heterogeneous"] == "configs/train_heterogeneous_ddp.yaml"
    assert defaults["model_parallel"] == "configs/train_model_parallel_kaggle.yaml"


def test_model_parallel_has_no_deep_trace():
    with pytest.raises(SystemExit):
        build_parser(get_strategy("model_parallel")).parse_args(["--trace", "deep"])
    build_parser(get_strategy("ddp")).parse_args(["--trace", "deep"])


@pytest.mark.parametrize("name,argv,flag", [
    ("heterogeneous", ["--precision", "amp"], "--precision"),
    ("heterogeneous", ["--batch-size", "8"], "--batch-size"),
    ("model_parallel", ["--inspect", "anomalies"], "--inspect")])
def test_an_option_the_strategy_cannot_honour_is_an_error(name, argv, flag, capsys):
    strategy = get_strategy(name)
    parser = build_parser(strategy)
    with pytest.raises(SystemExit) as exc:
        check_options(strategy, parser.parse_args(argv), parser)
    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("name", CANONICAL)
def test_options_that_every_strategy_supports_are_accepted(name, capsys):
    strategy = get_strategy(name)
    parser = build_parser(strategy)
    check_options(strategy, parser.parse_args(["--resume", "x.pt", "--layers", "plot"]), parser)


# ── CLI: nombres canónicos y alias ────────────────────────────────────────────

def test_cli_accepts_aliases_and_canonical_names():
    assert (build_train_cmd("model-parallel", "c.yaml") == build_train_cmd("model_parallel", "c.yaml")
            == build_train_cmd("mp", "c.yaml"))
    assert "scripts/train_heterogeneous_ddp.py" in build_train_cmd("hetero", "c.yaml")


def test_cli_forwards_the_new_options_after_the_old_ones():
    cmd = build_train_cmd("ddp", "c.yaml", layers=["plot"], batch_size=16, resume="run/",
                          inspect=["anomalies"])
    assert cmd[-8:] == ["--layers", "plot", "--batch-size", "16", "--resume", "run/",
                        "--inspect", "anomalies"]


def test_cli_rejects_a_trace_mode_the_strategy_lacks():
    with pytest.raises(ValueError, match="--trace deep"):
        build_train_cmd("model-parallel", "c.yaml", trace="deep")
    from typer.testing import CliRunner
    from src.cli import app
    out = CliRunner().invoke(app, ["train", "--strategy", "mp", "--trace", "deep", "--dry-run"])
    assert out.exit_code == 2 and "Traceback" not in out.output


def test_estimate_with_an_unknown_strategy_is_a_usage_error_not_a_traceback():
    from typer.testing import CliRunner
    from src.cli import app
    out = CliRunner().invoke(app, ["estimate", "--strategy", "foo"])
    assert out.exit_code == 2 and "unknown strategy" in out.output


def test_cli_rejects_what_the_strategy_cannot_do():
    with pytest.raises(ValueError, match="--batch-size"):
        build_train_cmd("heterogeneous", "c.yaml", batch_size=8)
    with pytest.raises(ValueError, match="--inspect"):
        build_train_cmd("model_parallel", "c.yaml", inspect=["anomalies"])


# ── Línea `Configuración:` con valores que el test de instantánea no cubre ────

def _ctx(name, cfg_training=None, **kw):
    from types import SimpleNamespace
    from src.training.tasks import get_task
    dataset = SimpleNamespace(name="toy", make_task=lambda: get_task("multilabel"))
    cfg = {"training": {"epochs": 3, "lr": 0.001, **(cfg_training or {})}, "model": {}}
    return LaunchContext(strategy=get_strategy(name), args=SimpleNamespace(split_block=None),
                         cfg=cfg, dataset=dataset, model_name="m", train_ds=range(90),
                         val_ds=range(10), **kw)


_BACKBONE = "freeze_backbone=false | backbone=timm"


def test_ddp_line_counts_the_gpus():
    ctx = _ctx("ddp", {"batch_size": 4, "precision": "amp"}, world_size=2)
    assert get_strategy("ddp").config_line(ctx) == (
        "modelo=m | batch=4/GPU (global=8, 2 GPUs) | epochs=3 | lr=0.001 | precision=amp | "
        f"loss=bce | train=90 | val=10 | dataset=toy | seed=None | {_BACKBONE}")


def test_heterogeneous_line_with_a_gpu_and_a_cpu_rank():
    ctx = _ctx("heterogeneous", {"pos_weight": [1.0]}, world_size=2,
               extras={"ranks_cfg": [{"batch_size": 64}, {"batch_size": 8}],
                       "train_sampler": range(80)})
    assert get_strategy("heterogeneous").config_line(ctx) == (
        "modelo=m | batch=GPU 64 / CPU 8 (global=72) | reparto=GPU 89% (~80 img) / "
        "CPU 11% (~10 img) | epochs=3 | lr=0.001 | loss=bce+pos_weight | train=90 | val=10 | "
        f"dataset=toy | seed=None | {_BACKBONE}")


def test_model_parallel_line_names_devices_and_split():
    ctx = _ctx("model_parallel", {"batch_size": 8}, extras={"devices": ["cuda:0", "cuda:1"]})
    ctx.args.split_block = 6
    assert get_strategy("model_parallel").config_line(ctx).startswith(
        "modelo=m | paralelismo=modelo(pipeline) | devices=cuda:0+cuda:1 | split_block=6 | "
        "batch=8 (global) | epochs=3")
