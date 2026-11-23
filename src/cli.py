"""ParaViT-Lab — unified command-line interface (``paravit <command>``).

One entry point for everything that runs in a terminal, where the compute lives:
launch trainings (any strategy), estimate before training (analytic), run a real
benchmark, evaluate on the test split, write a run report and open the dashboard. The web stays
read-only (visualisation); anything that needs a GPU is driven from here — the
same separation used by Weights & Biases / MLflow / TensorBoard.

    uv run paravit.py --help
    uv run paravit.py estimate --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2
    uv run paravit.py train   --strategy single --config configs/train.yaml
    uv run paravit.py train   --strategy ddp --n-gpus 2 --config configs/train_demo_ddp.yaml
    uv run paravit.py eval    --checkpoint checkpoints/local/single/vit_base_patch16_224/<sello>/best.pt --split test
    uv run paravit.py dashboard

(``uv run tfg.py`` keeps working as a backward-compatible alias.)

The argv builders (``build_*_cmd``) are pure and unit-tested; the Typer commands
just assemble them and run them (or print them with ``--dry-run``).
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
_DRY_RUN_HELP = "Print the command without running it."
console = Console()
app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="ParaViT-Lab — distributed Vision Transformers. Train, estimate, benchmark, evaluate and visualise.",
)

STRATEGIES = ("single", "ddp", "model-parallel", "heterogeneous")


# ── Pure command builders (unit-tested) ───────────────────────────────────────

def _torchrun(nproc: int, nnodes: int, node_rank: int,
              master_addr: str, master_port: int) -> list[str]:
    """torchrun prefix (via ``python -m torch.distributed.run`` for a stable path)."""
    base = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}"]
    if nnodes > 1:
        base += [f"--nnodes={nnodes}", f"--node_rank={node_rank}",
                 f"--master_addr={master_addr}", f"--master_port={master_port}"]
    return base


def build_train_cmd(strategy: str, config: str, *, model: str | None = None,
                    epochs: int | None = None, trace: str = "simple",
                    precision: str | None = None, layers: list[str] | None = None,
                    fn: list[str] | None = None, n_gpus: int = 2, nnodes: int = 1,
                    node_rank: int = 0, master_addr: str = "localhost",
                    master_port: int = 29500, monitor: bool = False,
                    seed: int | None = None, data_root: str | None = None) -> list[str]:
    """Build the exact command for a training strategy.

    single / model-parallel run a plain script; ddp / heterogeneous run under
    torchrun. Flags are only added where the target script accepts them.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")

    common = []
    if model:
        common += ["--model", model]
    if epochs:
        common += ["--epochs", str(epochs)]
    if seed is not None:
        common += ["--seed", str(seed)]
    if data_root:
        common += ["--data-root", data_root]
    common += ["--trace", trace]                      # every script accepts --trace
    if monitor:
        common += ["--monitor"]                       # every script accepts --monitor
    layers_fn = []
    if layers:
        layers_fn += ["--layers", *layers]
    if fn:
        layers_fn += ["--fn", *fn]
    # Tensor-core precision for the strategies whose loop honours it (single, ddp and
    # model-parallel all go through Trainer.train_epoch). Heterogeneous runs on gloo/CPU
    # with a hand-written fp32 loop, so it does not expose --precision.
    precision_arg = (["--precision", precision]
                     if precision and precision != "fp32" else [])

    if strategy == "single":
        return ([sys.executable, "scripts/train_single_gpu.py", "--config", config]
                + common + precision_arg + layers_fn)
    if strategy == "model-parallel":
        # model_parallel runs through the builder, so it honours --layers/--fn/--precision.
        return ([sys.executable, "scripts/train_model_parallel.py", "--config", config]
                + common + precision_arg + layers_fn)
    if strategy == "ddp":
        return (_torchrun(n_gpus, nnodes, node_rank, master_addr, master_port)
                + ["scripts/train_ddp.py", "--config", config]
                + common + precision_arg + layers_fn)
    # heterogeneous: one process per node, multi-node by definition
    return (_torchrun(1, nnodes, node_rank, master_addr, master_port)
            + ["scripts/train_heterogeneous_ddp.py", "--config", config] + common + layers_fn)


def build_benchmark_cmd(models: list[str] | None, batch_sizes: list[int] | None,
                          epochs: int, trace_modes: list[str] | None,
                          config: str | None = None, dataset_path: str | None = None,
                          precision: str = "fp32", compare_precision: bool = False,
                          convergence_study: bool = False, study_steps: int = 60,
                          nfs_factor: float = 1.0, device: int = 0,
                          train_size: int | None = None,
                          val_size: int | None = None) -> list[str]:
    """Build the benchmark.py command (the GPU benchmark / measurement)."""
    cmd = [sys.executable, "scripts/benchmark.py", "--epochs", str(epochs)]
    if config:
        cmd += ["--config", config]
    if models:
        cmd += ["--model", *models]
    if batch_sizes:
        cmd += ["--batch-sizes", *[str(b) for b in batch_sizes]]
    if trace_modes:
        cmd += ["--trace-modes", *trace_modes]
    if dataset_path:
        cmd += ["--dataset-path", dataset_path]
    if train_size is not None:
        cmd += ["--train-size", str(train_size)]
    if val_size is not None:
        cmd += ["--val-size", str(val_size)]
    if precision and precision != "fp32":
        cmd += ["--precision", precision]
    if compare_precision:
        cmd += ["--compare-precision"]
    if convergence_study:
        cmd += ["--convergence-study", "--study-steps", str(study_steps)]
    if not math.isclose(nfs_factor, 1.0):
        cmd += ["--nfs-factor", str(nfs_factor)]
    if device:
        cmd += ["--device", str(device)]
    return cmd


def build_eval_cmd(checkpoint: str | list[str], config: str, *, model: str | None = None,
                   split: str = "test", output: str | None = None,
                   metadata: str | None = None, batch_size: int | None = None,
                   max_batches: int | None = None,
                   save_predictions: str | None = None,
                   save_predictions_auto: bool = False,
                   data_root: str | None = None) -> list[str]:
    """Build the eval.py command (held-out evaluation). Several checkpoints = an ensemble
    (eval.py averages their probabilities)."""
    checkpoints = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    cmd = [sys.executable, "scripts/eval.py"]
    for ckpt in checkpoints:
        cmd += ["--checkpoint", ckpt]
    cmd += ["--config", config, "--split", split]
    if model:
        cmd += ["--model", model]
    if output:
        cmd += ["--output", output]
    if metadata:
        cmd += ["--metadata", metadata]
    if data_root:
        cmd += ["--data-root", data_root]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if max_batches:
        cmd += ["--max-batches", str(max_batches)]
    if save_predictions:
        cmd += ["--save-predictions", save_predictions]
    elif save_predictions_auto:
        cmd += ["--save-predictions"]       # no path: eval.py picks the run's log folder
    return cmd


def _abspath(path: str) -> str:
    """Paths given by the user are relative to WHERE they ran paravit, not to the repo root
    the scripts run from (``_run`` uses ``cwd=ROOT``)."""
    return str(Path(path).expanduser().resolve())


def build_predict_cmd(checkpoint: str | list[str], config: str, inputs: list[str], *,
                      model: str | None = None, out: str | None = None,
                      batch_size: int | None = None, device: str | None = None) -> list[str]:
    """Build the predict.py command (classify loose images or folders). Several checkpoints =
    an ensemble (predict.py averages their probabilities)."""
    checkpoints = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    cmd = [sys.executable, "scripts/predict.py"]
    for ckpt in checkpoints:
        cmd += ["--checkpoint", _abspath(ckpt)]
    cmd += ["--config", _abspath(config)]
    if model:
        cmd += ["--model", model]
    if out:
        cmd += ["--out", _abspath(out)]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if device:
        cmd += ["--device", device]
    return cmd + [_abspath(i) for i in inputs]


def build_compare_cmd(a: list[str], b: list[str], *, label_a: str = "A", label_b: str = "B",
                      group: str | None = None, n_boot: int = 2000, seed: int = 0,
                      markdown: bool = False) -> list[str]:
    """Build the compare_predictions.py command (paired comparison of two groups of runs)."""
    cmd = [sys.executable, "scripts/compare_predictions.py", "--a", *a, "--b", *b,
           "--label-a", label_a, "--label-b", label_b,
           "--n-boot", str(n_boot), "--seed", str(seed)]
    if group:
        cmd += ["--group", group]
    if markdown:
        cmd += ["--markdown"]
    return cmd


def build_select_ensemble_cmd(runs: list[str], *, val_name: str = "val_preds.npz",
                              test_name: str = "test_preds.npz", groups: list[str] | None = None,
                              method: str = "per-group", max_size: int = 8,
                              reference: list[str] | None = None, n_boot: int = 2000,
                              seed: int = 0, markdown: bool = False) -> list[str]:
    """Build the select_ensemble.py command (ensemble chosen on val, evaluated once on test)."""
    cmd = [sys.executable, "scripts/select_ensemble.py", "--runs", *runs,
           "--val-name", val_name, "--test-name", test_name, "--method", method,
           "--max-size", str(max_size), "--n-boot", str(n_boot), "--seed", str(seed)]
    for g in groups or []:
        cmd += ["--group", g]
    if reference:
        cmd += ["--reference", *reference]
    if markdown:
        cmd += ["--markdown"]
    return cmd


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], dry_run: bool) -> None:
    pretty = " ".join(cmd)
    console.print(Panel(pretty, title="command", border_style="cyan", expand=False))
    if dry_run:
        console.print("[yellow]--dry-run: not executed.[/]")
        return
    raise SystemExit(subprocess.run(cmd, cwd=str(ROOT)).returncode)


def resolve_dataset_n(dataset: str, dataset_size: int | None = None) -> int:
    """Map a dataset choice to a train-image count. ``full``/``subset`` are the
    project's two canonical sizes; an explicit ``dataset_size`` overrides them."""
    from src.performance_model import N_FULL_TRAIN, N_SUBSET_TRAIN
    if dataset_size:
        return int(dataset_size)
    return N_FULL_TRAIN if str(dataset).lower().startswith("full") else N_SUBSET_TRAIN


def _csv(s: str | None) -> list[str] | None:
    """Parse a comma-separated option (e.g. ``32,64`` or ``plot,confusion``)."""
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


# ── Commands ─────────────────────────────────────────────────────────────────

@app.command()
def train(
    strategy: str = typer.Option("single", help=f"One of {', '.join(STRATEGIES)}."),
    config: str = typer.Option("configs/train.yaml", help="YAML config (paths, model, batch…)."),
    model: Optional[str] = typer.Option(None, help="Override the model (any timm ID)."),
    epochs: Optional[int] = typer.Option(None, help="Override the number of epochs."),
    seed: Optional[int] = typer.Option(None, help="Override training.seed (repetitions)."),
    data_root: Optional[str] = typer.Option(None, help="Override data.root (dataset location)."),
    trace: str = typer.Option("simple", help="off | simple | deep."),
    precision: Optional[str] = typer.Option(None, help="fp32 | tf32 | amp | bf16 (single, ddp and model-parallel)."),
    layers: Optional[str] = typer.Option(None, help="Comma-separated: plot,confusion,batch-monitor,hooks."),
    fn: Optional[str] = typer.Option(None, help="Comma-separated: timing,energy."),
    n_gpus: int = typer.Option(2, help="GPUs per node (ddp)."),
    nnodes: int = typer.Option(1, help="Number of nodes (multi-node ddp / heterogeneous)."),
    node_rank: int = typer.Option(0, help="This node's rank (multi-node)."),
    master_addr: str = typer.Option("localhost", help="Rendez-vous host (multi-node)."),
    master_port: int = typer.Option(29500, help="Rendez-vous port (multi-node)."),
    monitor: bool = typer.Option(False, "--monitor",
                                 help="Per-sample predictions, embeddings, Grad-CAM and "
                                      "layer stats every epoch (see the web Inspector)."),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """Launch a training run with the chosen distribution strategy."""
    cmd = build_train_cmd(strategy, config, model=model, epochs=epochs, trace=trace,
                          precision=precision, layers=_csv(layers), fn=_csv(fn), n_gpus=n_gpus,
                          nnodes=nnodes, node_rank=node_rank, master_addr=master_addr,
                          master_port=master_port, monitor=monitor, seed=seed,
                          data_root=data_root)
    _run(cmd, dry_run)


@app.command(name="estimate")
def predict(
    model: str = typer.Option("vit_base_patch16_224", help="Model (timm ID)."),
    gpu: str = typer.Option("Tesla T4", help="GPU name, e.g. 'Tesla V100', 'RTX 3060 Ti'."),
    strategy: str = typer.Option("single", help="single | ddp | model_parallel | heterogeneous."),
    n_gpus: int = typer.Option(1, help="Number of GPUs."),
    batch: int = typer.Option(96, help="Global batch size."),
    precision: str = typer.Option("fp32", help="fp32 | amp | tf32 | bf16."),
    dataset: str = typer.Option("subset", help="Dataset size: full (237,871) | subset (5,000)."),
    dataset_size: Optional[int] = typer.Option(None, help="Custom train-image count (overrides --dataset)."),
    epochs: int = typer.Option(15, help="Number of epochs."),
    disk: str = typer.Option("ssd", help="ssd | nvme | hdd | nfs."),
    in_chans: int = typer.Option(3, help="Input channels (≠3 profiles the model instead of using its table entry)."),
    num_classes: int = typer.Option(19, help="Output classes (negligible effect on cost; head only)."),
    img_size: int = typer.Option(224, help="Input resolution (≠224 profiles the model instead of using its table entry)."),
) -> None:
    """Analytic estimate of time, memory, cost and quality for a config — no GPU, just formulas.

    Models outside the curated table are profiled automatically (params and FLOPs
    measured with torch's FLOP counter on the meta device; no weights, no GPU)."""
    n = resolve_dataset_n(dataset, dataset_size)
    _show_prediction(model, gpu, strategy, n_gpus, batch, precision, n, epochs, disk,
                     in_chans=in_chans, num_classes=num_classes, img_size=img_size)


def _show_prediction(model: str, gpu: str, strategy: str, n_gpus: int, batch: int,
                     precision: str, dataset_size: int, epochs: int, disk: str,
                     in_chans: int = 3, num_classes: int = 19, img_size: int = 224) -> None:
    """Render the analytic prediction in full: a headline, the time and memory
    formulas with the values plugged in, expected quality, scaling and cloud cost.
    Shared by `predict` and `menu`."""
    from src.model_profile import resolve_model_spec
    from src.performance_model import (
        predict as _predict, predict_quality, gpu_spec,
        estimate_rc, estimate_rio, precision_factor,
        BYTES_PER_PARAM_GRAD_OPT, CUDA_OVERHEAD_GB, EVAL_POWER_FRACTION)
    nfs = disk == "nfs"
    strat = strategy.replace("-", "_")
    ms, gs = resolve_model_spec(model, in_chans, num_classes, img_size), gpu_spec(gpu)
    if ms is None:
        console.print(f"[red]Unknown model '{model}': not in the table and timm cannot build it.[/]")
        return
    if gs is None:
        console.print(f"[red]Unknown GPU '{gpu}': not in the table (see GPU_TABLE in "
                      "src/performance_model.py); a similar card is never assumed.[/]")
        return
    p = _predict(strat, model, gpu, n_gpus=n_gpus, dataset_size=dataset_size,
                 batch=batch, precision=precision, epochs=epochs, disk_type=disk, nfs=nfs,
                 model=ms)
    if p is None:
        console.print("[red]Unknown model or GPU spec.[/]")
        return
    q = predict_quality(model, dataset_size=dataset_size, epochs=epochs,
                        calibrated=ms.calibrated)

    # ── Headline ────────────────────────────────────────────────────────────────
    fit = "[green]fits[/]" if p.fits_in_memory else "[red]OOM[/]"
    sp = f" · {p.speedup:.2f}× ({p.efficiency*100:.0f}% eff)" if strat != "single" else ""
    console.print(Panel(
        f"[bold]{model}[/] · {gpu} · {strategy} ×{n_gpus} · {precision} · "
        f"{dataset_size:,} imgs/epoch\n"
        f"[bold]{_fmt_secs(p.time_per_epoch_train_s)}[/]/epoch{sp}  ·  "
        f"{p.vram_per_gpu_gb:.1f} GB/GPU {fit}  ·  "
        f"total [bold]{_fmt_secs(p.time_total_train_s)}[/] for {epochs} ep  ·  "
        f"~{p.energy_per_epoch_wh:.1f} Wh/epoch ([bold]{p.energy_total_wh:.0f} Wh[/] total)  ·  "
        f"bottleneck [bold]{p.bottleneck}[/]",
        title="Prediction", border_style="cyan", expand=False))

    # ── Time / epoch: the master formula with the values plugged in ──────────────
    tt = Table(title="Time / epoch = max(compute, I/O) + sync", border_style="cyan")
    tt.add_column("Term", style="bold"); tt.add_column("Value", justify="right")
    tt.add_column("Formula", style="dim")
    tt.add_row("Compute", f"{p.t_compute_s:.0f} s", "N / (π · r_c · n_gpus)")
    tt.add_row("Data I/O", f"{p.t_io_s:.0f} s", "N / r_io  (fixed, shared disk)")
    tt.add_row("Grad sync", f"{p.t_sync_s:.1f} s", "(8·P / β) · n_batches")
    tt.add_row("Time/epoch", _fmt_secs(p.time_per_epoch_train_s),
               f"max({p.t_compute_s:.0f}, {p.t_io_s:.0f}) + {p.t_sync_s:.1f} → "
               f"[bold]{p.bottleneck}[/]-bound")
    console.print(tt)

    # ── VRAM: the memory formula with the values plugged in ──────────────────────
    if ms and gs:
        amp = precision in ("amp", "bf16")
        wo = BYTES_PER_PARAM_GRAD_OPT * ms.params_m * 1e6 / 1e9 + (2 * ms.params_m * 1e6 / 1e9 if amp else 0)
        act = ms.act_gb_per_img * p.batch_per_gpu * (0.6 if amp else 1.0)
        mt = Table(title="VRAM / GPU = weights+grad+optimizer + activations + overhead",
                   border_style="cyan")
        mt.add_column("Term", style="bold"); mt.add_column("Value", justify="right")
        mt.add_column("Formula", style="dim")
        mt.add_row("Weights+grad+Adam", f"{wo:.2f} GB",
                   f"16 B × {ms.params_m:.0f}M" + (" + fp16 copy" if amp else ""))
        mt.add_row("Activations", f"{act:.2f} GB",
                   f"{p.batch_per_gpu} × {ms.act_gb_per_img:.3f} GB/img" + (" × 0.6" if amp else "")
                   + ("" if ms.calibrated else " (estimated by rule)"))
        mt.add_row("CUDA overhead", f"{CUDA_OVERHEAD_GB:.2f} GB", "context + cudnn")
        mt.add_row("Total", f"{p.vram_per_gpu_gb:.1f} GB",
                   f"vs {gs.vram_gb:.0f} GB → {'fits' if p.fits_in_memory else 'OOM'} · "
                   f"max batch {p.recommended_batch}")
        console.print(mt)

    # ── Energy: power × time, with calibrated effective power ──────────────────────
    _n_work = round(p.power_total_w / p.avg_power_w) if p.avg_power_w else n_gpus
    _p_src = "measured calibration" if p.power_calibrated else "TDP fallback (not measured)"
    et = Table(title="Energy / epoch = power × time", border_style="cyan")
    et.add_column("Term", style="bold"); et.add_column("Value", justify="right")
    et.add_column("Formula", style="dim")
    et.add_row("Power / GPU", f"{p.avg_power_w:.0f} W", _p_src)
    et.add_row("Total power", f"{p.power_total_w:.0f} W",
               f"{p.avg_power_w:.0f} W × {_n_work} working GPU(s)")
    et.add_row("Energy train", f"{p.energy_train_wh:.2f} Wh",
               f"{p.power_total_w:.0f} W × {p.time_per_epoch_train_s:.0f} s / 3600")
    et.add_row("Energy eval", f"{p.energy_eval_wh:.2f} Wh",
               f"{EVAL_POWER_FRACTION:g} × {p.power_total_w:.0f} W × {p.time_per_epoch_eval_s:.0f} s / 3600")
    et.add_row("Energy/epoch", f"{p.energy_per_epoch_wh:.2f} Wh", "train + eval")
    et.add_row(f"Energy total ({epochs} ep)", f"{p.energy_total_wh:.1f} Wh", "energy/epoch × epochs")
    console.print(et)

    # ── Expected quality ─────────────────────────────────────────────────────────
    if q is not None:
        qt = Table(title="Expected quality (empirical prior — not a measurement)",
                   border_style="cyan", show_header=False)
        qt.add_column("k", style="bold"); qt.add_column("v")
        qt.add_row("Expected Val F1", f"{q.expected_best_f1:.3f} ± {q.band:.3f}")
        qt.add_row("Best epoch ≈", str(q.best_epoch))
        qt.add_row("Early stop ≈", str(q.early_stop_epoch))
        qt.add_row("Confidence", str(q.confidence)
                   + ("" if ms.calibrated else " (architecture outside the curated table)"))
        console.print(qt)

    # ── Scaling 1→8 GPUs (data-parallel) ─────────────────────────────────────────
    if strat in ("ddp", "heterogeneous"):
        sct = Table(title="Scaling (predicted)", border_style="cyan")
        sct.add_column("GPUs", justify="right"); sct.add_column("Speedup", justify="right")
        sct.add_column("Efficiency", justify="right"); sct.add_column("Time/epoch", justify="right")
        for ng in (1, 2, 4, 8):
            pn = _predict(strat, model, gpu, n_gpus=ng, dataset_size=dataset_size, batch=batch,
                          precision=precision, epochs=1, disk_type=disk, nfs=nfs, model=ms)
            if pn:
                sct.add_row(str(ng), f"{pn.speedup:.2f}×", f"{pn.efficiency*100:.0f}%",
                            _fmt_secs(pn.time_per_epoch_train_s))
        console.print(sct)

    # ── Cloud cost ───────────────────────────────────────────────────────────────
    try:
        from src.cloud_cost import estimate_costs
        total_h = p.time_total_train_s / 3600 * (n_gpus if strat in ("ddp", "heterogeneous") else 1)
        rows = [r for r in estimate_costs(total_h, gpu) if r["usd_per_hour"] > 0][:5]
        if rows:
            ct = Table(title=f"Cloud cost · {total_h:.1f} GPU-hours", border_style="cyan")
            ct.add_column("Provider"); ct.add_column("GPU")
            ct.add_column("$/h", justify="right"); ct.add_column("Cost", justify="right")
            for r in rows:
                ct.add_row(r["provider"], r["gpu"], f"${r['usd_per_hour']:.2f}", f"${r['cost_usd']:.2f}")
            console.print(ct)
    except Exception:
        pass

    # ── Assumptions behind the formulas ──────────────────────────────────────────
    if ms and gs:
        rc = estimate_rc(ms, gs, precision)
        rio = estimate_rio(disk, nfs)
        console.print(
            f"[dim]Assumptions: r_c ≈ {rc:.0f} img/s/GPU (incl. precision ×"
            f"{precision_factor(gs, precision):.2f}) · r_io ≈ {rio:.0f} img/s "
            f"({disk}{'+NFS' if nfs else ''}) · {ms.params_m:.0f}M params · MFU 0.17. "
            f"Calibrate with a measured throughput in the Benchmark CLI.[/]")
    if p.calibrated:
        console.print("[green]Calibrated with a measured throughput.[/]")
    for note in p.notes:
        console.print(f"[yellow]•[/] {note}")


@app.command(name="benchmark")
def benchmark(
    model: Optional[str] = typer.Option(None, help="Comma-separated model(s), e.g. vit_base_patch16_224,resnet50."),
    batch_sizes: Optional[str] = typer.Option(None, help="Comma-separated batch sizes, e.g. 32,64."),
    epochs: int = typer.Option(30, help="Epochs for the time estimate."),
    trace_modes: Optional[str] = typer.Option(None, help="Comma-separated: off,simple,deep."),
    config: Optional[str] = typer.Option(None, help="YAML config (for dataset sizes/paths)."),
    dataset_path: Optional[str] = typer.Option(None, help="Dataset path to measure real I/O."),
    precision: str = typer.Option("fp32", help="Benchmark precision (Tensor-core switch)."),
    compare_precision: bool = typer.Option(False, "--compare-precision", help="FP32 vs Tensor."),
    convergence_study: bool = typer.Option(False, "--convergence-study", help="Real mini-training."),
    study_steps: int = typer.Option(60, help="Mini-training steps."),
    nfs_factor: float = typer.Option(1.0, help="NFS latency factor (Verode ≈ 1.3)."),
    device: int = typer.Option(0, help="CUDA device index."),
    train_size: Optional[int] = typer.Option(None, help="Train images (default: the config's dataset)."),
    val_size: Optional[int] = typer.Option(None, help="Val images (default: the config's dataset)."),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """Empirical benchmark on this machine — measures real throughput, memory and scaling."""
    _bs = [int(x) for x in (_csv(batch_sizes) or [])] or None
    cmd = build_benchmark_cmd(_csv(model), _bs, epochs, _csv(trace_modes), config=config,
                                dataset_path=dataset_path, precision=precision,
                                compare_precision=compare_precision,
                                convergence_study=convergence_study, study_steps=study_steps,
                                nfs_factor=nfs_factor, device=device,
                                train_size=train_size, val_size=val_size)
    _run(cmd, dry_run)


@app.command()
def eval(
    checkpoint: List[str] = typer.Option(
        ..., help="Path to the .pt checkpoint. Repeat it to evaluate an ensemble "
                  "(the probabilities of the checkpoints are averaged)."),
    config: str = typer.Option("configs/train.yaml", help="YAML config (paths, model)."),
    model: Optional[str] = typer.Option(None, help="Override the model."),
    split: str = typer.Option("test", help="test | val."),
    output: Optional[str] = typer.Option(None, help="CSV path for per-class results."),
    metadata: Optional[str] = typer.Option(None, help="Override metadata.parquet."),
    data_root: Optional[str] = typer.Option(None, help="Override data.root (dataset location)."),
    batch_size: Optional[int] = typer.Option(None, help="Batch size."),
    max_batches: Optional[int] = typer.Option(None, help="Stop after N batches (sanity)."),
    save_predictions: Optional[str] = typer.Option(
        None, "--save-predictions", help=".npz path for the per-sample predictions."),
    save_predictions_auto: bool = typer.Option(
        False, "--save-predictions-auto",
        help="Save the predictions as test_preds_<ts>.npz in the run's log folder "
             "(the web Inspector reads them)."),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """Evaluate a checkpoint (or an ensemble of several) on the held-out test (or val) split."""
    cmd = build_eval_cmd(checkpoint, config, model=model, split=split, output=output,
                         metadata=metadata, batch_size=batch_size, max_batches=max_batches,
                         data_root=data_root, save_predictions=save_predictions,
                         save_predictions_auto=save_predictions_auto)
    _run(cmd, dry_run)


@app.command("predict")
def predict_images(
    inputs: List[str] = typer.Argument(..., help="Image files or folders (recursive)."),
    checkpoint: List[str] = typer.Option(
        ..., help="Path to the .pt checkpoint or run folder. Repeat it for an ensemble "
                  "(the probabilities of the checkpoints are averaged)."),
    config: str = typer.Option("configs/train.yaml", help="YAML config (dataset, model)."),
    model: Optional[str] = typer.Option(None, help="Override the model (default: the checkpoint's)."),
    out: Optional[str] = typer.Option(None, help="CSV path with the probability of every class."),
    batch_size: Optional[int] = typer.Option(None, help="Images per batch."),
    device: Optional[str] = typer.Option(None, help="cpu | cuda (default: cuda if available)."),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """Classify loose images (or folders of them) with a trained checkpoint, outside training.
    Uses the dataset's own evaluation transform; several checkpoints form an ensemble."""
    if device is not None and device not in ("cpu", "cuda"):
        raise typer.BadParameter("must be cpu or cuda", param_hint="--device")
    cmd = build_predict_cmd(checkpoint, config, inputs, model=model, out=out,
                            batch_size=batch_size, device=device)
    _run(cmd, dry_run)


@app.command("compare-preds")
def compare_preds(
    a: List[str] = typer.Option(..., "--a", help="Predictions .npz of group A. Repeat it per run."),
    b: List[str] = typer.Option(..., "--b", help="Predictions .npz of group B (reference). Repeat it per run."),
    label_a: str = typer.Option("A", help="Name of group A in the tables."),
    label_b: str = typer.Option("B", help="Name of group B in the tables."),
    group: Optional[str] = typer.Option(None, help="Subgroup variable of the .npz (e.g. data_provider)."),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples."),
    seed: int = typer.Option(0, help="Bootstrap seed."),
    markdown: bool = typer.Option(False, "--markdown", help="Markdown tables (decimal comma) for docs/."),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """Compare two groups of runs (e.g. 3 seeds of a variant vs 3 of the base) from their saved
    test predictions: mean ± std, ensemble and paired bootstrap of the difference. No GPU."""
    cmd = build_compare_cmd(a, b, label_a=label_a, label_b=label_b, group=group,
                            n_boot=n_boot, seed=seed, markdown=markdown)
    _run(cmd, dry_run)


@app.command("select-ensemble")
def select_ensemble(
    runs: List[str] = typer.Option(..., "--runs", help="Run folder with val and test predictions. Repeat it per run."),
    val_name: str = typer.Option("val_preds.npz", help="Validation predictions file in each run folder."),
    test_name: str = typer.Option("test_preds.npz", help="Test predictions file in each run folder."),
    group: Optional[List[str]] = typer.Option(None, "--group", help="PATTERN=GROUP on run folder names (e.g. 'swin_t*=swin_t'). Repeat it; default: every run its own group."),
    method: str = typer.Option("per-group", help="greedy | per-group."),
    max_size: int = typer.Option(8, help="Greedy: maximum ensemble size."),
    reference: Optional[List[str]] = typer.Option(None, "--reference", help="Run folder of the reference ensemble. Repeat it."),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples."),
    seed: int = typer.Option(0, help="Bootstrap seed."),
    markdown: bool = typer.Option(False, "--markdown", help="Markdown tables (decimal comma) for docs/."),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """Choose an ensemble of runs on validation only and evaluate it once on test (optionally
    against a reference ensemble with the paired bootstrap). From saved predictions; no GPU."""
    if method not in ("greedy", "per-group"):
        raise typer.BadParameter("must be greedy or per-group", param_hint="--method")
    cmd = build_select_ensemble_cmd(runs, val_name=val_name, test_name=test_name, groups=group,
                                    method=method, max_size=max_size, reference=reference,
                                    n_boot=n_boot, seed=seed, markdown=markdown)
    _run(cmd, dry_run)


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port for the Streamlit server."),
    dry_run: bool = typer.Option(False, "--dry-run", help=_DRY_RUN_HELP),
) -> None:
    """Open the read-only web dashboard (Overview · Live · Inspector · Run results ·
    Compare · Estimate/Benchmark · Dataset). Plan a config without running with `paravit estimate`."""
    cmd = [sys.executable, "-m", "streamlit", "run", "src/web/app.py", "--server.port", str(port)]
    _run(cmd, dry_run)


def _run_row(r) -> dict:
    """Summarise a RunInfo for the `runs` table: the best validation value of the run's main
    metric (QWK for PANDA, AUC for MedMNIST, F1 otherwise; from the epoch CSV) and the
    held-out test value of that metric (from the run's own test_<ts>.csv, if present)."""
    from src.web.run_meta import resolve_metric
    metric = r.main_metric
    best_val = None
    epochs = None
    if r.epoch_csv_path and Path(r.epoch_csv_path).exists():
        try:
            import pandas as pd
            df = pd.read_csv(r.epoch_csv_path)
            epochs = len(df)
            metric = resolve_metric(metric, df)
            if metric.column in df.columns and df[metric.column].notna().any():
                best_val = float(df[metric.column].max())
        except Exception:
            pass
    test_val = None
    if getattr(r, "test_csv_paths", None):
        try:
            from src.web.eval_parser import parse_eval_csv
            _, agg = parse_eval_csv(r.test_csv_paths[0])
            # the eval CSV carries f1_* for every dataset, auc always, qwk for ordinal ones
            keys = {"qwk": ("qwk",), "auc": ("auc",)}.get(metric.key, ("f1_opt", "f1_t05"))
            test_val = next((agg[k] for k in keys if k in agg), None)
        except Exception:
            pass
    return {"label": r.label, "env": r.env, "mode": r.mode,
            "precision": r.precision or "fp32", "epochs": epochs,
            "metric": metric.name, "best_val": best_val, "test": test_val}


@app.command()
def runs(
    env: Optional[str] = typer.Option(None, help="Filter by environment (local/verode/kaggle)."),
    limit: int = typer.Option(20, help="Maximum number of rows."),
) -> None:
    """List the training runs in logs/ (best validation and held-out test value of each run's
    main metric: QWK for PANDA, AUC for MedMNIST, F1 otherwise)."""
    from src.web.run_registry import discover_runs
    found = discover_runs(ROOT)
    if env:
        found = [r for r in found if r.env == env]
    found = found[:limit]
    if not found:
        console.print("[yellow]No runs found in logs/.[/]")
        return
    t = Table(title=f"Runs ({len(found)})", border_style="cyan")
    for col in ("Run", "Env", "Mode", "Prec", "Epochs", "Metric", "Best Val", "Test"):
        t.add_column(col)
    for r in found:
        d = _run_row(r)
        t.add_row(d["label"], d["env"], d["mode"], d["precision"],
                  str(d["epochs"] or "—"), d["metric"],
                  f"{d['best_val']:.4f}" if d["best_val"] is not None else "—",
                  f"[bold green]{d['test']:.4f}[/]" if d["test"] is not None else "—")
    console.print(t)

@app.command()
def report(
    run: str = typer.Argument(..., help="An events_*.jsonl, a run folder, or a DDMMYYYY_HHMMSS "
                                        "stamp (searched under logs/)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
                                          help="Output file (default: report_<stamp>.html next "
                                               "to the run's events file)."),
    epoch: Optional[int] = typer.Option(None, help="Inspected epoch to evaluate (default: the "
                                                   "last one)."),
    n_boot: int = typer.Option(1000, min=1, help="Bootstrap resamples for the confidence "
                                                 "intervals."),
    seed: int = typer.Option(0, help="Bootstrap random seed (written in the report)."),
    bins: int = typer.Option(15, min=1, help="Calibration bins."),
    errors: int = typer.Option(8, min=0, help="Worst-error thumbnails to show."),
) -> None:
    """Write a self-contained HTML report of a run (summary, training, evaluation with
    confidence intervals, calibration, dynamics and a TRIPOD+AI checklist). Reads logs only:
    no GPU, nothing is trained. Needs --monitor data for the evaluation sections."""
    from src.reporting import NoInspection, RunNotFound, write_report
    try:
        out = write_report(run, output, root=ROOT, epoch=epoch, n_boot=n_boot, seed=seed,
                           bins=bins, k_errors=errors)
    except (RunNotFound, NoInspection) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    console.print(f"[green]Report written:[/] {out}")


@app.command()
def menu() -> None:
    """Interactive guided menu — pick an action and fill in the parameters.

    Each action asks the common parameters and then offers an "Advanced options"
    gate that exposes every flag the equivalent command accepts. The interactive
    walkthrough lives in src/cli_menu.py (single-responsibility split)."""
    from src.cli_menu import run_menu
    run_menu()


if __name__ == "__main__":
    app()
