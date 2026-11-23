"""Common launcher for every training strategy.

``scripts/train_*.py`` are 5-line wrappers around :func:`main`: the flow (config, seeds, data,
builder, console summary, ``Configuración:`` line, fit, shutdown) lives here once, and what
changes between strategies — launch mode, process group and device, DataLoaders, the builder's own
``with_*``, the text of the config line, own options — comes from the
:class:`~src.training.strategies.StrategySpec` of the strategy.

Every strategy accepts the same options (the ones a strategy cannot honour fail with an explicit
error instead of being ignored) plus the ones its spec adds.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

from src.training.config_overrides import add_common_overrides, prepare_config
from src.training.strategies import LaunchContext, StrategySpec, get_strategy

_PRECISIONS = ["fp32", "tf32", "amp", "bf16"]
_METRICS = ["loss", "f1", "accuracy", "precision_recall"]
_STAMP_FORMAT = "%d%m%Y_%H%M%S"


def build_parser(strategy: StrategySpec) -> argparse.ArgumentParser:
    """Parser with every common option plus the ones of ``strategy``."""
    p = argparse.ArgumentParser(description=strategy.description)
    p.add_argument("--config", type=str, default=strategy.default_config)
    p.add_argument("--epochs", type=int, help="Override training.epochs")
    p.add_argument("--model", type=str, help="Override model name (any timm ID or registered backbone)")
    p.add_argument("--precision", choices=_PRECISIONS, default=None,
                   help="Numeric precision = the practical Tensor-core switch. fp32 (CUDA cores) | "
                        "tf32/amp/bf16 (Tensor cores). amp=FP16 mixed precision (Volta+), "
                        "tf32/bf16 (Ampere+).")
    p.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size "
                   "(per GPU in DDP)")
    add_common_overrides(p)
    p.add_argument("--monitor", action="store_true",
                   help="Prediction/layer monitor: per-sample predictions, embeddings, Grad-CAM "
                        "and layer stats every epoch (docs/inspeccion.md)")
    p.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT",
                   help="Path to a checkpoint .pt file (or run folder) to resume training from")
    p.add_argument("--trace", choices=list(strategy.trace_choices), default="simple",
                   help="Logging controller mode (distributed: applies only to rank 0)")
    p.add_argument("--inspect", nargs="*", default=None,
                   choices=["model-summary", "grad-monitor", "batch-table", "anomalies"],
                   help="Modular inspection features (activates DeepTracingDecorator; "
                        "distributed: rank 0 only)")
    p.add_argument("--layers", nargs="*", default=[],
                   choices=["plot", "hooks", "confusion", "batch-monitor"],
                   help="Stackable aspect decorators (distributed: rank 0 only)")
    p.add_argument("--fn", nargs="*", choices=["timing", "energy"], default=[],
                   help="Python @ decorators to apply to train_epoch / eval_epoch")
    p.add_argument("--batch-log-every", type=int, default=None, metavar="N",
                   help="Log batch metrics every N batches when --layers batch-monitor is "
                        "active (default 1; use 50-100 to reduce the CSV size)")
    p.add_argument("--metrics", nargs="*", choices=_METRICS, default=list(_METRICS),
                   help="Metric reporters (only active without --inspect; distributed: rank 0 "
                        "only). Pass --metrics with no args to disable all.")
    if strategy.add_arguments:
        strategy.add_arguments(p)
    return p


def check_options(strategy: StrategySpec, args: argparse.Namespace,
                  parser: argparse.ArgumentParser) -> None:
    """Fail on an option this strategy cannot honour, saying why (never ignore it silently)."""
    for dest, reason in strategy.unsupported.items():
        if getattr(args, dest, None) is not None:
            flag = "--" + dest.replace("_", "-")
            parser.error(f"{flag} no está disponible en la estrategia '{strategy.name}': {reason}")


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.model:
        cfg["model"]["name"] = args.model      # also what validate_config and the log line see
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.precision:
        cfg["training"]["precision"] = args.precision


def _seed_everything(ctx: LaunchContext) -> None:
    """Fix every RNG BEFORE creating data and model (deterministic head init, shuffling,
    augmentations and mixup). Distributed runs add the rank, so ranks do not repeat each other."""
    from src.training.reproducibility import make_generator, seed_worker, set_seed
    seed = ctx.seed = ctx.cfg["training"].get("seed")
    if seed is None:
        return
    set_seed(int(seed) + ctx.rank)
    if ctx.strategy.loader_rng:
        ctx.loader_generator = make_generator(int(seed) + ctx.rank)
        ctx.worker_init = seed_worker
    if ctx.is_main:
        note = "(+rank, run determinista)" if ctx.strategy.distributed else "(run determinista)"
        print(f"Semilla     : {seed} {note}")


def _shared_timestamp(strategy: StrategySpec, rank: int) -> str:
    """The same timestamp on every rank, so logs and checkpoints get consistent names."""
    stamp = [datetime.now().strftime(_STAMP_FORMAT) if rank == 0 else ""]
    if strategy.distributed:
        import torch.distributed as dist
        dist.broadcast_object_list(stamp, src=0)
    return stamp[0]


def _gated(ctx: LaunchContext, value, off):
    """Decorators, logs and reporters belong to rank 0; the other ranks get ``off``."""
    return value if ctx.is_main else off


def _build_trainer(ctx: LaunchContext):
    args = ctx.args
    trace = _gated(ctx, args.trace, "off")
    layers = _gated(ctx, args.layers or [], [])
    metrics = _gated(ctx, args.metrics or [], [])
    inspect = _gated(ctx, args.inspect, None)
    builder = (ctx.strategy.make_builder(ctx)
               .with_trace(trace)
               .with_layers(*layers)
               .with_fn(*(args.fn or []))
               .with_metrics(*metrics))
    if args.model:
        builder = builder.with_model(args.model)
    if inspect is not None:
        builder = builder.with_inspect(*inspect)
    if args.batch_log_every is not None:
        builder = builder.with_batch_log_every(args.batch_log_every)
    if args.monitor:
        builder = builder.with_monitor()
    return builder.build(), trace, layers, metrics, inspect


def _resume(ctx: LaunchContext, trainer, checkpoint: str) -> None:
    """Restore model, optimizer, scheduler and best-so-far from ``checkpoint`` and train only the
    remaining epochs. Every rank loads the same file, so the replicas start identical (the file
    is written by rank 0: on several machines it must be reachable from all of them)."""
    inner = trainer
    while hasattr(inner, "_trainer"):               # walk the decorator stack to the core Trainer
        inner = inner._trainer
    if not hasattr(inner, "load_checkpoint"):
        print("WARN: trainer no soporta load_checkpoint, ignorando --resume")
        return
    resumed_epoch = inner.load_checkpoint(checkpoint).get("epoch", 0)
    remaining = ctx.cfg["training"]["epochs"] - resumed_epoch
    if ctx.is_main:
        print(f"Resumiendo desde epoch {resumed_epoch} — quedan {remaining} epochs")
    ctx.cfg["training"]["epochs"] = remaining


def _inspect_label(inspect, trace: str):
    if inspect is not None:
        return sorted(inspect)
    return "todas" if trace == "deep" else "ninguna"


def _print_summary(ctx: LaunchContext, trace, layers, metrics, inspect) -> None:
    """Console summary (rank 0): what was built. None of these lines is parsed by the web."""
    use_deep = trace == "deep" or inspect is not None
    where = f"  |  rank {ctx.rank}/{ctx.world_size}" if ctx.strategy.distributed else ""
    print(f"Dispositivo : {ctx.device}{where}")
    for line in (ctx.strategy.describe(ctx) if ctx.strategy.describe else []):
        print(line)
    print(f"Modelo      : {ctx.model_name}")
    print(f"Traza       : {trace}" + (" [DeepTracingDecorator]" if use_deep else ""))
    if "inspect" not in ctx.strategy.unsupported:
        print(f"Inspect     : {_inspect_label(inspect, trace)}")
    print(f"Capas       : {layers or 'ninguna'}")
    print(f"Decoradores@: {ctx.args.fn or 'ninguno'}")
    print(f"Métricas    : {metrics or 'ninguna (DeepTracingDecorator las gestiona)'}")


def main(strategy_name: str, argv: list[str] | None = None) -> None:
    """Launch a training run with ``strategy_name`` (``argv`` defaults to ``sys.argv[1:]``)."""
    from src.data.registry import dataset_spec_from_config

    strategy = get_strategy(strategy_name)
    parser = build_parser(strategy)
    args = parser.parse_args(argv)
    check_options(strategy, args, parser)

    cfg = prepare_config(args)
    _apply_overrides(cfg, args)
    ctx = LaunchContext(strategy=strategy, args=args, cfg=cfg)
    strategy.init_env(ctx)
    _seed_everything(ctx)
    ctx.timestamp = _shared_timestamp(strategy, ctx.rank)

    # ── Data ──────────────────────────────────────────────────────────────────
    ctx.dataset = dataset_spec_from_config(cfg)
    ctx.train_ds = ctx.dataset.dataset(cfg["data"], "train")
    ctx.val_ds = ctx.dataset.dataset(cfg["data"], "val")
    ctx.model_name = cfg["model"]["name"]
    if ctx.is_main:
        print(f"Train: {ctx.n_train} patches | Val: {ctx.n_val} patches")
    strategy.make_loaders(ctx)

    # ── Trainer stack (builder) ───────────────────────────────────────────────
    trainer, trace, layers, metrics, inspect = _build_trainer(ctx)
    if args.resume:
        _resume(ctx, trainer, args.resume)
    if ctx.is_main:
        _print_summary(ctx, trace, layers, metrics, inspect)
        # The run's configuration in the log (the web shows it in Run → Info)
        logging.getLogger("trainer").info(f"Configuración: {strategy.config_line(ctx)}")

    # ── Training ──────────────────────────────────────────────────────────────
    trainer.fit(ctx.train_loader, ctx.val_loader, epochs=cfg["training"]["epochs"])

    if strategy.distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
