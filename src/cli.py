"""ParaViT-Lab — unified command-line interface (``paravit <command>``).

One entry point for everything that runs in a terminal, where the compute lives:
launch trainings (any strategy), estimate before training (analytic), run a real
benchmark, evaluate on the test split, and open the dashboard. The web stays
read-only (visualisation); anything that needs a GPU is driven from here — the
same separation used by Weights & Biases / MLflow / TensorBoard.

    uv run paravit.py --help
    uv run paravit.py estimate --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2
    uv run paravit.py train   --strategy single --config configs/train.yaml
    uv run paravit.py train   --strategy ddp --n-gpus 2 --config configs/train_demo_ddp.yaml
    uv run paravit.py eval    --checkpoint checkpoints/local/checkpoint_epoch_009.pt --split test
    uv run paravit.py dashboard

(``uv run tfg.py`` keeps working as a backward-compatible alias.)

The argv builders (``build_*_cmd``) are pure and unit-tested; the Typer commands
just assemble them and run them (or print them with ``--dry-run``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
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
                    master_port: int = 29500) -> list[str]:
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
    common += ["--trace", trace]                      # every script accepts --trace
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
                          nfs_factor: float = 1.0, device: int = 0) -> list[str]:
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
    if precision and precision != "fp32":
        cmd += ["--precision", precision]
    if compare_precision:
        cmd += ["--compare-precision"]
    if convergence_study:
        cmd += ["--convergence-study", "--study-steps", str(study_steps)]
    if nfs_factor != 1.0:
        cmd += ["--nfs-factor", str(nfs_factor)]
    if device:
        cmd += ["--device", str(device)]
    return cmd


def build_eval_cmd(checkpoint: str, config: str, *, model: str | None = None,
                   split: str = "test", output: str | None = None,
                   metadata: str | None = None, batch_size: int | None = None,
                   max_batches: int | None = None) -> list[str]:
    """Build the eval.py command (held-out evaluation)."""
    cmd = [sys.executable, "scripts/eval.py", "--checkpoint", checkpoint,
           "--config", config, "--split", split]
    if model:
        cmd += ["--model", model]
    if output:
        cmd += ["--output", output]
    if metadata:
        cmd += ["--metadata", metadata]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if max_batches:
        cmd += ["--max-batches", str(max_batches)]
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
    trace: str = typer.Option("simple", help="off | simple | deep."),
    precision: Optional[str] = typer.Option(None, help="fp32 | tf32 | amp | bf16 (single-GPU)."),
    layers: Optional[str] = typer.Option(None, help="Comma-separated: plot,confusion,batch-monitor,hooks."),
    fn: Optional[str] = typer.Option(None, help="Comma-separated: timing,energy."),
    n_gpus: int = typer.Option(2, help="GPUs per node (ddp)."),
    nnodes: int = typer.Option(1, help="Number of nodes (multi-node ddp / heterogeneous)."),
    node_rank: int = typer.Option(0, help="This node's rank (multi-node)."),
    master_addr: str = typer.Option("localhost", help="Rendez-vous host (multi-node)."),
    master_port: int = typer.Option(29500, help="Rendez-vous port (multi-node)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without running it."),
) -> None:
    """Launch a training run with the chosen distribution strategy."""
    cmd = build_train_cmd(strategy, config, model=model, epochs=epochs, trace=trace,
                          precision=precision, layers=_csv(layers), fn=_csv(fn), n_gpus=n_gpus,
                          nnodes=nnodes, node_rank=node_rank, master_addr=master_addr,
                          master_port=master_port)
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
) -> None:
    """Analytic estimate of time, memory, cost and quality for a config — no GPU, just formulas."""
    n = resolve_dataset_n(dataset, dataset_size)
    _show_prediction(model, gpu, strategy, n_gpus, batch, precision, n, epochs, disk)


def _show_prediction(model: str, gpu: str, strategy: str, n_gpus: int, batch: int,
                     precision: str, dataset_size: int, epochs: int, disk: str) -> None:
    """Render the analytic prediction in full: a headline, the time and memory
    formulas with the values plugged in, expected quality, scaling and cloud cost.
    Shared by `predict` and `menu`."""
    from src.performance_model import (
        predict as _predict, predict_quality, model_spec, gpu_spec,
        estimate_rc, estimate_rio, precision_factor,
        BYTES_PER_PARAM_GRAD_OPT, CUDA_OVERHEAD_GB, EVAL_POWER_FRACTION)
    nfs = disk == "nfs"
    strat = strategy.replace("-", "_")
    p = _predict(strat, model, gpu, n_gpus=n_gpus, dataset_size=dataset_size,
                 batch=batch, precision=precision, epochs=epochs, disk_type=disk, nfs=nfs)
    if p is None:
        console.print("[red]Unknown model or GPU spec.[/]")
        return
    ms, gs = model_spec(model), gpu_spec(gpu)
    q = predict_quality(model, dataset_size=dataset_size, epochs=epochs)

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
                   f"{p.batch_per_gpu} × {ms.act_gb_per_img:.3f} GB/img" + (" × 0.6" if amp else ""))
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
        qt.add_row("Confidence", str(q.confidence))
        console.print(qt)

    # ── Scaling 1→8 GPUs (data-parallel) ─────────────────────────────────────────
    if strat in ("ddp", "heterogeneous"):
        sct = Table(title="Scaling (predicted)", border_style="cyan")
        sct.add_column("GPUs", justify="right"); sct.add_column("Speedup", justify="right")
        sct.add_column("Efficiency", justify="right"); sct.add_column("Time/epoch", justify="right")
        for ng in (1, 2, 4, 8):
            pn = _predict(strat, model, gpu, n_gpus=ng, dataset_size=dataset_size, batch=batch,
                          precision=precision, epochs=1, disk_type=disk, nfs=nfs)
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
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without running it."),
) -> None:
    """Empirical benchmark on this machine — measures real throughput, memory and scaling."""
    _bs = [int(x) for x in (_csv(batch_sizes) or [])] or None
    cmd = build_benchmark_cmd(_csv(model), _bs, epochs, _csv(trace_modes), config=config,
                                dataset_path=dataset_path, precision=precision,
                                compare_precision=compare_precision,
                                convergence_study=convergence_study, study_steps=study_steps,
                                nfs_factor=nfs_factor, device=device)
    _run(cmd, dry_run)


@app.command()
def eval(
    checkpoint: str = typer.Option(..., help="Path to the .pt checkpoint."),
    config: str = typer.Option("configs/train.yaml", help="YAML config (paths, model)."),
    model: Optional[str] = typer.Option(None, help="Override the model."),
    split: str = typer.Option("test", help="test | val."),
    output: Optional[str] = typer.Option(None, help="CSV path for per-class results."),
    metadata: Optional[str] = typer.Option(None, help="Override metadata.parquet."),
    batch_size: Optional[int] = typer.Option(None, help="Batch size."),
    max_batches: Optional[int] = typer.Option(None, help="Stop after N batches (sanity)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without running it."),
) -> None:
    """Evaluate a checkpoint on the held-out test (or val) split."""
    cmd = build_eval_cmd(checkpoint, config, model=model, split=split, output=output,
                         metadata=metadata, batch_size=batch_size, max_batches=max_batches)
    _run(cmd, dry_run)


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port for the Streamlit server."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without running it."),
) -> None:
    """Open the read-only web dashboard (Overview · Run results · Compare ·
    Performance · Dataset). Plan a config without running with `paravit estimate`."""
    cmd = [sys.executable, "-m", "streamlit", "run", "src/web/app.py", "--server.port", str(port)]
    _run(cmd, dry_run)


def _run_row(r) -> dict:
    """Summarise a RunInfo for the `runs` table: best Val F1 (from the epoch CSV)
    and the held-out Test F1 (from a test_*.csv, if present)."""
    best_val_f1 = None
    epochs = None
    if r.epoch_csv_path and Path(r.epoch_csv_path).exists():
        try:
            import pandas as pd
            df = pd.read_csv(r.epoch_csv_path)
            epochs = len(df)
            if "val_f1" in df.columns and df["val_f1"].notna().any():
                best_val_f1 = float(df["val_f1"].max())
        except Exception:
            pass
    test_f1 = None
    if getattr(r, "test_csv_paths", None):
        try:
            from src.web.eval_parser import parse_eval_csv
            _, agg = parse_eval_csv(r.test_csv_paths[0])
            test_f1 = agg.get("f1_opt", agg.get("f1_t05"))
        except Exception:
            pass
    return {"label": r.label, "env": r.env, "mode": r.mode,
            "precision": r.precision or "fp32", "epochs": epochs,
            "best_val_f1": best_val_f1, "test_f1": test_f1}


@app.command()
def runs(
    env: Optional[str] = typer.Option(None, help="Filter by environment (local/verode/kaggle)."),
    limit: int = typer.Option(20, help="Maximum number of rows."),
) -> None:
    """List the training runs in logs/ (best Val F1, and held-out Test F1 if available)."""
    from src.web.run_registry import discover_runs
    found = discover_runs(ROOT)
    if env:
        found = [r for r in found if r.env == env]
    found = found[:limit]
    if not found:
        console.print("[yellow]No runs found in logs/.[/]")
        return
    t = Table(title=f"Runs ({len(found)})", border_style="cyan")
    for col in ("Run", "Env", "Mode", "Prec", "Epochs", "Best Val F1", "Test F1"):
        t.add_column(col)
    for r in found:
        d = _run_row(r)
        t.add_row(d["label"], d["env"], d["mode"], d["precision"],
                  str(d["epochs"] or "—"),
                  f"{d['best_val_f1']:.4f}" if d["best_val_f1"] is not None else "—",
                  f"[bold green]{d['test_f1']:.4f}[/]" if d["test_f1"] is not None else "—")
    console.print(t)

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
