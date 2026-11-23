"""StrategySpec — everything the launcher needs to know about one training strategy.

Each strategy (single GPU, DDP, heterogeneous DDP, model parallel) is described once by a
:class:`StrategySpec` and registered here, like :class:`src.data.spec.DatasetSpec` does for the
datasets. The common launcher (:mod:`src.training.launch`), the ``paravit train`` command and the
interactive menu ask the spec — how it is launched, how it initialises the process and picks the
device, how it builds the DataLoaders, what the builder needs and what its ``Configuración:`` line
says — instead of keeping one script per strategy with its own copy of the whole flow.

Adding a strategy = a ``Trainer`` subclass, a ``with_*`` in the builder, and a spec at the end of
this module (plus a ~5-line ``scripts/train_<name>.py`` wrapper if people launch it by path).

This module is import-cheap on purpose (the CLI imports it): torch, the dataset registry and the
builder are imported lazily inside the hooks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

_CUDA_0 = "cuda:0"
_CPU = "cpu"
_ALL_TRACES = ("off", "simple", "deep")


@dataclass
class LaunchContext:
    """State of one launched process, filled in step by step by the common flow and the hooks."""
    strategy: "StrategySpec"
    args: Any                      # argparse.Namespace (common + strategy options)
    cfg: dict
    dataset: Any = None            # DatasetSpec
    device: Any = None             # torch.device where the (core of the) model lives
    rank: int = 0
    world_size: int = 1
    seed: int | None = None
    timestamp: str = ""
    model_name: str = ""
    loader_generator: Any = None
    worker_init: Callable | None = None
    train_ds: Any = None
    val_ds: Any = None
    train_loader: Any = None
    val_loader: Any = None
    extras: dict = field(default_factory=dict)   # what only one strategy needs (devices, ranks…)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def n_train(self) -> int:
        return len(self.train_ds)

    @property
    def n_val(self) -> int:
        return len(self.val_ds)


@dataclass(frozen=True)
class StrategySpec:
    name: str                          # canonical name = the folder under logs/ and checkpoints/
    description: str
    script: str                        # scripts/train_<x>.py: what the CLI launches
    default_config: str
    init_env: Callable[[LaunchContext], None]       # process group + device (sets ctx.device/rank…)
    make_loaders: Callable[[LaunchContext], None]   # sets ctx.train_loader / ctx.val_loader
    make_builder: Callable[[LaunchContext], Any]    # TrainingSessionBuilder with its own with_*
    config_line: Callable[[LaunchContext], str]     # text after "Configuración:" (web contract)
    aliases: tuple[str, ...] = ()
    launcher: str = "python"           # "python" (plain process) | "torchrun"
    nproc: int | None = None           # torchrun processes per node; None = one per GPU (--n-gpus)
    distributed: bool = False          # process group: rank offset in the seed, shared timestamp…
    loader_rng: bool = True            # seeded DataLoader generator + worker_init_fn
    default_workers: int = 0           # data.num_workers when the config does not say
    trace_choices: tuple[str, ...] = _ALL_TRACES
    #: option dest -> why this strategy cannot honour it (an error, never silently ignored)
    unsupported: Mapping[str, str] = field(default_factory=dict)
    add_arguments: Callable[[Any], None] | None = None   # options only this strategy has
    describe: Callable[[LaunchContext], list[str]] | None = None   # extra console summary lines

    def __post_init__(self):
        if not self.name or self.name != _normalise(self.name):
            raise ValueError(f"strategy name {self.name!r} must be lowercase with underscores")
        if self.launcher not in ("python", "torchrun"):
            raise ValueError(f"{self.name}: launcher must be 'python' or 'torchrun'")
        if self.launcher == "torchrun" and not self.distributed:
            raise ValueError(f"{self.name}: a torchrun strategy is distributed")

    @property
    def torchrun(self) -> bool:
        return self.launcher == "torchrun"


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, StrategySpec] = {}
_ALIASES: dict[str, str] = {}


def _normalise(name: str) -> str:
    """``Model-Parallel`` and ``model_parallel`` are the same strategy."""
    return name.strip().lower().replace("-", "_")


def register_strategy(spec: StrategySpec) -> StrategySpec:
    """Add ``spec`` (and its aliases) to the registry (idempotent for the same object)."""
    current = _REGISTRY.get(spec.name)
    if current is not None and current is not spec:
        raise ValueError(f"strategy {spec.name!r} is already registered")
    for alias in spec.aliases:
        taken = _ALIASES.get(_normalise(alias))
        if taken not in (None, spec.name):
            raise ValueError(f"alias {alias!r} already belongs to strategy {taken!r}")
    _REGISTRY[spec.name] = spec
    for alias in spec.aliases:
        _ALIASES[_normalise(alias)] = spec.name
    return spec


def available_strategies() -> list[str]:
    """Canonical names, in registration order."""
    return list(_REGISTRY)


def get_strategy(name: str) -> StrategySpec:
    """The spec called ``name`` (canonical name or alias; ``-`` and ``_`` are equivalent)."""
    key = _normalise(name)
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: "
                         f"{available_strategies()}") from None


# ── Environment hooks: process group, rank and device ─────────────────────────

def _require_torchrun_env() -> None:
    missing = [k for k in ("RANK", "WORLD_SIZE") if k not in os.environ]
    if missing:
        raise SystemExit(
            f"ERROR: falta {', '.join(missing)} en el entorno: esta estrategia es distribuida y "
            "se lanza con torchrun, p. ej.  torchrun --nproc_per_node=2 scripts/train_ddp.py ... "
            "(o con `paravit train --strategy ddp`).")


def _local_rank(rank: int) -> int:
    return int(os.environ.get("LOCAL_RANK", rank))


def _cuda_device_of(local_rank: int):
    """The GPU of this process: the node-local index (multi-node with 1 GPU per node → cuda:0)."""
    import torch
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    return device


def _init_single(ctx: LaunchContext) -> None:
    import torch
    ctx.device = torch.device("cuda" if torch.cuda.is_available() else _CPU)


def _init_ddp(ctx: LaunchContext) -> None:
    import torch
    import torch.distributed as dist
    _require_torchrun_env()
    backend = ctx.cfg.get("distributed", {}).get("backend", "nccl")
    if backend == "nccl" and torch.cuda.is_available():
        # One process per GPU: NCCL does not allow two ranks on the same GPU.
        ctx.device = _cuda_device_of(_local_rank(int(os.environ["RANK"])))
        dist.init_process_group(backend="nccl", device_id=ctx.device)
    else:
        ctx.device = torch.device(_CPU)
        dist.init_process_group(backend="gloo")
    ctx.rank, ctx.world_size = dist.get_rank(), dist.get_world_size()


def _hetero_rank_batch(ctx: LaunchContext, rank: int) -> int:
    """Batch of ``rank``: its own ``distributed.ranks[].batch_size``, else ``training.batch_size``."""
    ranks_cfg = ctx.extras["ranks_cfg"]
    if rank < len(ranks_cfg) and "batch_size" in ranks_cfg[rank]:
        return ranks_cfg[rank]["batch_size"]
    # Lazy fallback: the heterogeneous config may have no training.batch_size at all.
    return ctx.cfg.get("training", {}).get("batch_size", 32)


def _init_heterogeneous(ctx: LaunchContext) -> None:
    import torch
    import torch.distributed as dist
    _require_torchrun_env()
    dist_cfg = ctx.cfg.get("distributed", {})
    ctx.extras["backend"] = dist_cfg.get("backend", "gloo")
    ctx.extras["ranks_cfg"] = ranks_cfg = dist_cfg.get("ranks", [])
    dist.init_process_group(backend=ctx.extras["backend"])
    ctx.rank, ctx.world_size = dist.get_rank(), dist.get_world_size()
    # The device of this rank comes from the config (GPU node + CPU nodes)
    if ranks_cfg and ctx.rank < len(ranks_cfg):
        wanted = ranks_cfg[ctx.rank].get("device", _CPU)
    else:
        wanted = "cuda" if torch.cuda.is_available() else _CPU
    if wanted == "cuda" and torch.cuda.is_available():
        ctx.device = _cuda_device_of(_local_rank(ctx.rank))
    else:
        ctx.device = torch.device(_CPU)
    ctx.extras["batch_size"] = _hetero_rank_batch(ctx, ctx.rank)
    if ranks_cfg and ctx.rank < len(ranks_cfg):
        ctx.extras["weights"] = [r.get("compute_weight", 1) for r in ranks_cfg]
    else:
        ctx.extras["weights"] = [1] * ctx.world_size


def _pick_devices(arg: str | None) -> list[str]:
    """The two stages: ``--devices`` or what the machine has (2 GPUs, 1 GPU twice, or CPU)."""
    import torch
    if arg:
        return [d.strip() for d in arg.split(",")]
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n >= 2:
        return [_CUDA_0, "cuda:1"]
    if n == 1:
        return [_CUDA_0, _CUDA_0]
    return [_CPU, _CPU]


def _init_model_parallel(ctx: LaunchContext) -> None:
    import torch
    ctx.extras["devices"] = devices = _pick_devices(ctx.args.devices)
    ctx.device = torch.device(devices[-1])      # logits, labels and loss live on the last stage


# ── DataLoader hooks ──────────────────────────────────────────────────────────

def _num_workers(ctx: LaunchContext) -> int:
    return ctx.cfg["data"].get("num_workers", ctx.strategy.default_workers)


def _make_plain_loaders(ctx: LaunchContext) -> None:
    """Whole dataset in one process: shuffled train, ordered val (single, model parallel)."""
    from torch.utils.data import DataLoader
    bs, nw = ctx.cfg["training"]["batch_size"], _num_workers(ctx)
    ctx.train_loader = DataLoader(
        ctx.train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
        generator=ctx.loader_generator, worker_init_fn=ctx.worker_init)
    ctx.val_loader = DataLoader(
        ctx.val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
        worker_init_fn=ctx.worker_init)


def _val_sampler(ctx: LaunchContext):
    """drop_last=False keeps every validation sample, so the metrics are unbiased."""
    from torch.utils.data import DistributedSampler
    return DistributedSampler(ctx.val_ds, num_replicas=ctx.world_size, rank=ctx.rank,
                              shuffle=False, drop_last=False)


def _make_ddp_loaders(ctx: LaunchContext) -> None:
    """Each rank reads its shard: drop_last=True on train keeps equal-sized batches."""
    from torch.utils.data import DataLoader, DistributedSampler
    from src.training.reproducibility import sampler_seed
    train_sampler = DistributedSampler(
        ctx.train_ds, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, drop_last=True,
        seed=sampler_seed(ctx.seed))
    bs, nw = ctx.cfg["training"]["batch_size"], _num_workers(ctx)
    ctx.train_loader = DataLoader(
        ctx.train_ds, batch_size=bs, sampler=train_sampler, num_workers=nw, pin_memory=True,
        worker_init_fn=ctx.worker_init, generator=ctx.loader_generator)
    ctx.val_loader = DataLoader(
        ctx.val_ds, batch_size=bs, sampler=_val_sampler(ctx), num_workers=nw, pin_memory=True,
        worker_init_fn=ctx.worker_init)


def _make_heterogeneous_loaders(ctx: LaunchContext) -> None:
    """Each rank takes a share of the train set proportional to its ``compute_weight`` and
    its own batch size (the GPU a big one, the CPU a small one)."""
    from torch.utils.data import DataLoader
    from src.training.heterogeneous_sampler import HeterogeneousDistributedSampler
    from src.training.reproducibility import sampler_seed
    ctx.extras["train_sampler"] = train_sampler = HeterogeneousDistributedSampler(
        ctx.train_ds, weights=ctx.extras["weights"], rank=ctx.rank, world_size=ctx.world_size,
        shuffle=True, drop_last=True, seed=sampler_seed(ctx.seed))
    val_sampler = _val_sampler(ctx)
    ctx.extras["val_sampler"] = val_sampler
    bs, nw = ctx.extras["batch_size"], _num_workers(ctx)
    pin = ctx.device.type == "cuda"
    ctx.train_loader = DataLoader(ctx.train_ds, batch_size=bs, sampler=train_sampler,
                                  num_workers=nw, pin_memory=pin)
    ctx.val_loader = DataLoader(ctx.val_ds, batch_size=bs, sampler=val_sampler,
                                num_workers=nw, pin_memory=pin)


# ── Builder hooks: the strategy's own with_* ──────────────────────────────────

def _builder_class():
    from src.training.builder import TrainingSessionBuilder
    return TrainingSessionBuilder


def _build_single(ctx: LaunchContext):
    return _builder_class()(ctx.cfg, ctx.device, ctx.timestamp)


def _build_ddp(ctx: LaunchContext):
    return _builder_class()(ctx.cfg, ctx.device, ctx.timestamp, rank=ctx.rank,
                            world_size=ctx.world_size, distributed=True)


def _build_heterogeneous(ctx: LaunchContext):
    # HeterogeneousDDPTrainer wraps the model with DDP once and normalises the gradients by the
    # real global batch; this rank's batch size goes into its config copy.
    local_bs = ctx.extras["batch_size"]
    cfg_copy = {**ctx.cfg, "training": {**ctx.cfg["training"], "batch_size": local_bs}}
    return (_builder_class()(cfg_copy, ctx.device, ctx.timestamp, rank=ctx.rank,
                             world_size=ctx.world_size, distributed=True)
            .with_heterogeneous_ddp(local_bs)
            .with_output_mode("ddp_hetero"))


def _build_model_parallel(ctx: LaunchContext):
    return (_builder_class()(ctx.cfg, ctx.device, ctx.timestamp)
            .with_model_parallel(ctx.extras["devices"], ctx.args.split_block))


# ── `Configuración:` line (contract with src/web and src/run_manifest.py) ─────

def _config_tail(ctx: LaunchContext, *, precision: bool = True, loss: str | None = None,
                 ousm: bool = True) -> str:
    """``epochs=… | … | seed=… | backbone…``: the part every strategy shares."""
    from src.models.backbones import backbone_log_fields
    from src.training.losses import loss_log_label, ousm_log_field
    t = ctx.cfg["training"]
    parts = [f"epochs={t['epochs']}", f"lr={t['lr']}"]
    if precision:
        parts.append(f"precision={t.get('precision', 'fp32')}")
    parts.append(f"loss={loss or loss_log_label(t, ctx.dataset.make_task())}")
    parts += [f"train={ctx.n_train}", f"val={ctx.n_val}", f"dataset={ctx.dataset.name}",
              f"seed={ctx.seed}", backbone_log_fields(ctx.cfg["model"], ctx.model_name)]
    return " | ".join(parts) + (ousm_log_field(t) if ousm else "")


def _line_single(ctx: LaunchContext) -> str:
    bs = ctx.cfg["training"]["batch_size"]
    return f"modelo={ctx.model_name} | batch={bs}/GPU (global={bs}) | {_config_tail(ctx)}"


def _line_ddp(ctx: LaunchContext) -> str:
    bs = ctx.cfg["training"]["batch_size"]
    return (f"modelo={ctx.model_name} | batch={bs}/GPU "
            f"(global={bs * ctx.world_size}, {ctx.world_size} GPUs) | {_config_tail(ctx)}")


def _line_model_parallel(ctx: LaunchContext) -> str:
    bs = ctx.cfg["training"]["batch_size"]
    return (f"modelo={ctx.model_name} | paralelismo=modelo(pipeline) | "
            f"devices={'+'.join(ctx.extras['devices'])} | "
            f"split_block={ctx.args.split_block or 'auto'} | batch={bs} (global) | "
            f"{_config_tail(ctx)}")


def _line_heterogeneous(ctx: LaunchContext) -> str:
    # Different batch per rank (GPU vs CPU) and the real split of the data between them.
    ranks_cfg = ctx.extras["ranks_cfg"]
    bs_gpu = _hetero_rank_batch(ctx, 0)
    bs_cpu = _hetero_rank_batch(ctx, 1) if len(ranks_cfg) > 1 else None
    n_tot = ctx.n_train
    n_gpu = len(ctx.extras["train_sampler"])        # samples of rank 0 (the GPU)
    n_cpu = n_tot - n_gpu
    # The trainer always trains with BCE (summed), with pos_weight if the config has one
    loss = "bce+pos_weight" if ctx.cfg["training"].get("pos_weight") is not None else "bce"
    return (f"modelo={ctx.model_name} | "
            f"batch=GPU {bs_gpu} / CPU {bs_cpu if bs_cpu is not None else '—'} "
            f"(global={bs_gpu + (bs_cpu or 0)}) | "
            f"reparto=GPU {n_gpu / n_tot * 100:.0f}% (~{n_gpu} img) / "
            f"CPU {n_cpu / n_tot * 100:.0f}% (~{n_cpu} img) | "
            f"{_config_tail(ctx, precision=False, loss=loss, ousm=False)}")


# ── Console summary and own options ───────────────────────────────────────────

def _describe_ddp(ctx: LaunchContext) -> list[str]:
    bs = ctx.cfg["training"]["batch_size"]
    return [f"Batch/GPU   : {bs}  → global batch: {bs * ctx.world_size}"]


def _describe_heterogeneous(ctx: LaunchContext) -> list[str]:
    total = sum(_hetero_rank_batch(ctx, r) for r in range(ctx.world_size))
    return [f"DDP heterogéneo — {ctx.world_size} ranks | backend: {ctx.extras['backend']}",
            f"Batch global: {total} | pesos: {ctx.extras['weights']}",
            f"Batch rank 0: {ctx.extras['batch_size']} | muestras: "
            f"{len(ctx.extras['train_sampler'])} train / {len(ctx.extras['val_sampler'])} val"]


def _describe_model_parallel(ctx: LaunchContext) -> list[str]:
    devices = ctx.extras["devices"]
    real = "reparto real en 2 dispositivos" if len(set(devices)) > 1 else "un solo dispositivo — demo"
    return [f"Devices     : {devices} ({real})"]


def _add_model_parallel_arguments(parser) -> None:
    parser.add_argument("--split-block", type=int, default=None,
                        help="Block index where stage 0 ends (default: half the blocks)")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated device list, e.g. 'cuda:0,cuda:1'. "
                             "Auto-detected if omitted.")


# ── The four strategies ───────────────────────────────────────────────────────

register_strategy(StrategySpec(
    name="single", description="Single-GPU training",
    script="scripts/train_single_gpu.py", default_config="configs/train.yaml",
    aliases=("single_gpu",),
    init_env=_init_single, make_loaders=_make_plain_loaders, make_builder=_build_single,
    config_line=_line_single))

register_strategy(StrategySpec(
    name="ddp", description="Distributed training (multi-GPU, multi-node DDP)",
    script="scripts/train_ddp.py", default_config="configs/train_cluster.yaml",
    launcher="torchrun", distributed=True,
    init_env=_init_ddp, make_loaders=_make_ddp_loaders, make_builder=_build_ddp,
    config_line=_line_ddp, describe=_describe_ddp))

register_strategy(StrategySpec(
    name="heterogeneous", description="Heterogeneous DDP training (GPU + CPU nodes)",
    script="scripts/train_heterogeneous_ddp.py",
    default_config="configs/train_heterogeneous_ddp.yaml",
    aliases=("heterogeneous_ddp", "hetero"),
    launcher="torchrun", nproc=1, distributed=True, loader_rng=False,
    init_env=_init_heterogeneous, make_loaders=_make_heterogeneous_loaders,
    make_builder=_build_heterogeneous, config_line=_line_heterogeneous,
    describe=_describe_heterogeneous,
    unsupported={
        "precision": "su bucle manual corre en fp32 (gloo/CPU)",
        "batch_size": "el lote se define por rank en distributed.ranks[].batch_size "
                      "(o con --set training.batch_size=N si la config no lista ranks)",
    }))

register_strategy(StrategySpec(
    name="model_parallel", description="Model-parallel (pipeline) training across two devices",
    script="scripts/train_model_parallel.py",
    default_config="configs/train_model_parallel_kaggle.yaml",
    aliases=("model-parallel", "mp"),
    default_workers=4, trace_choices=("off", "simple"),
    init_env=_init_model_parallel, make_loaders=_make_plain_loaders,
    make_builder=_build_model_parallel, config_line=_line_model_parallel,
    describe=_describe_model_parallel, add_arguments=_add_model_parallel_arguments,
    unsupported={"inspect": "la inspección profunda necesita un modelo en un solo dispositivo"}))
