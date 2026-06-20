"""Unified command-line interface — ``tfg <command>``.

One entry point for everything that runs in a terminal, where the compute lives:
launch trainings (any strategy), predict before training, run the feasibility
benchmark, evaluate on the test split, and open the dashboard. The web stays
read-only (visualisation); anything that needs a GPU is driven from here — the
same separation used by Weights & Biases / MLflow / TensorBoard.

    uv run tfg.py --help
    uv run tfg.py predict --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2
    uv run tfg.py train   --strategy single --config configs/train.yaml
    uv run tfg.py train   --strategy ddp --n-gpus 2 --config configs/train_demo_ddp.yaml
    uv run tfg.py eval    --checkpoint checkpoints/local/checkpoint_epoch_009.pt --split test
    uv run tfg.py dashboard

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
    help="TFG — distributed Transformers. Train, predict, evaluate and visualise.",
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

    if strategy == "single":
        cmd = [sys.executable, "scripts/train_single_gpu.py", "--config", config] + common
        if precision and precision != "fp32":         # only single exposes --precision
            cmd += ["--precision", precision]
        return cmd + layers_fn
    if strategy == "model-parallel":
        # model_parallel takes config/model/epochs/trace only (no layers/fn).
        return [sys.executable, "scripts/train_model_parallel.py", "--config", config] + common
    if strategy == "ddp":
        return (_torchrun(n_gpus, nnodes, node_rank, master_addr, master_port)
                + ["scripts/train_ddp.py", "--config", config] + common + layers_fn)
    # heterogeneous: one process per node, multi-node by definition
    return (_torchrun(1, nnodes, node_rank, master_addr, master_port)
            + ["scripts/train_heterogeneous_ddp.py", "--config", config] + common + layers_fn)


def build_feasibility_cmd(models: list[str] | None, batch_sizes: list[int] | None,
                          epochs: int, trace_modes: list[str] | None,
                          config: str | None = None, dataset_path: str | None = None,
                          precision: str = "fp32", compare_precision: bool = False,
                          convergence_study: bool = False, study_steps: int = 60,
                          nfs_factor: float = 1.0, device: int = 0) -> list[str]:
    """Build the check_feasibility.py command (the GPU benchmark / measurement)."""
    cmd = [sys.executable, "scripts/check_feasibility.py", "--epochs", str(epochs)]
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


@app.command()
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
    """Predict time, memory, cost and quality for a config — no GPU, just formulas."""
    n = resolve_dataset_n(dataset, dataset_size)
    _show_prediction(model, gpu, strategy, n_gpus, batch, precision, n, epochs, disk)


def _show_prediction(model: str, gpu: str, strategy: str, n_gpus: int, batch: int,
                     precision: str, dataset_size: int, epochs: int, disk: str) -> None:
    """Render the analytic prediction as a rich table (shared by `predict` and `menu`)."""
    from src.performance_model import predict as _predict, predict_quality
    nfs = disk == "nfs"
    strat = strategy.replace("-", "_")
    p = _predict(strat, model, gpu, n_gpus=n_gpus, dataset_size=dataset_size,
                 batch=batch, precision=precision, epochs=epochs, disk_type=disk, nfs=nfs)
    if p is None:
        console.print("[red]Unknown model or GPU spec.[/]")
        return
    q = predict_quality(model, dataset_size=dataset_size, epochs=epochs)

    t = Table(title=f"Prediction · {model} · {gpu} · {strategy} ×{n_gpus} · {precision}",
              show_header=False, border_style="cyan")
    t.add_column("metric", style="bold")
    t.add_column("value")
    t.add_row("Train / epoch", _fmt_secs(p.time_per_epoch_train_s))
    t.add_row("Total train", _fmt_secs(p.time_total_train_s))
    if strat != "single":
        t.add_row("Speedup", f"{p.speedup:.2f}×  (efficiency {p.efficiency*100:.0f}%)")
    t.add_row("Bottleneck", p.bottleneck)
    t.add_row("VRAM / GPU", f"{p.vram_per_gpu_gb:.1f} GB  "
              + ("[green]fits[/]" if p.fits_in_memory else "[red]OOM[/]"))
    t.add_row("Max batch that fits", str(p.recommended_batch))
    if q is not None:
        t.add_row("Expected Val F1", f"{q.expected_best_f1:.3f} ± {q.band:.3f}  "
                  f"(best ep ≈ {q.best_epoch}, confidence {q.confidence})")
    console.print(t)
    try:
        from src.cloud_cost import estimate_costs
        total_h = p.time_total_train_s / 3600 * (n_gpus if strat in ("ddp", "heterogeneous") else 1)
        rows = [r for r in estimate_costs(total_h, gpu) if r["usd_per_hour"] > 0][:3]
        if rows:
            console.print("[dim]Cheapest paid cloud:[/] " + "  ·  ".join(
                f"{r['provider']} {r['gpu']} ${r['cost_usd']:.2f}" for r in rows))
    except Exception:
        pass
    for note in p.notes:
        console.print(f"[yellow]•[/] {note}")


@app.command()
def feasibility(
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
    """Run the feasibility benchmark on this machine (measures real throughput)."""
    _bs = [int(x) for x in (_csv(batch_sizes) or [])] or None
    cmd = build_feasibility_cmd(_csv(model), _bs, epochs, _csv(trace_modes), config=config,
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
    """Open the web dashboard (read-only visualisation of runs and predictions)."""
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


def _pick(label: str, options: list[str], default: str | None = None,
          allow_other: bool = False, none_label: str | None = None) -> str | None:
    """Numbered picker — shows the valid values so the user doesn't have to guess.

    Returns the chosen string, or None if ``none_label`` is selected. With
    ``allow_other`` the user can type a custom value (e.g. any timm model id)."""
    from rich.prompt import IntPrompt, Prompt
    rows, mapping, idx = [], {}, 1
    if none_label is not None:
        rows.append(f"  [cyan]{idx}[/] {none_label}"); mapping[idx] = ("none", None); idx += 1
    for o in options:
        rows.append(f"  [cyan]{idx}[/] {o}"); mapping[idx] = ("val", o); idx += 1
    if allow_other:
        rows.append(f"  [cyan]{idx}[/] other…"); mapping[idx] = ("other", None); idx += 1
    console.print(f"[bold]{label}[/]\n" + "\n".join(rows))
    di = next((k for k, v in mapping.items() if v == ("val", default)), 1)
    n = IntPrompt.ask("  #", default=di)
    kind, val = mapping.get(n, mapping[di])
    if kind == "none":
        return None
    if kind == "other":
        return Prompt.ask("  Value")
    return val


def _pick_model(label: str, default: str = "vit_base_patch16_224",
                allow_from_config: bool = False) -> str | None:
    """Model picker for the commands that accept any timm model. Shows the short
    list of models relevant to this project, plus a 'search all timm' option
    (timm has ~1300 models — too many to list) and a free-text 'other'."""
    from rich.prompt import IntPrompt, Prompt
    from src.performance_model import MODEL_TABLE
    rows, mapping, idx = [], {}, 1
    if allow_from_config:
        rows.append(f"  [cyan]{idx}[/] (from config)"); mapping[idx] = ("none", None); idx += 1
    for m in MODEL_TABLE:
        rows.append(f"  [cyan]{idx}[/] {m}"); mapping[idx] = ("val", m); idx += 1
    rows.append(f"  [cyan]{idx}[/] search all timm models…"); mapping[idx] = ("search", None); idx += 1
    rows.append(f"  [cyan]{idx}[/] other (type any id)…"); mapping[idx] = ("other", None); idx += 1
    console.print(f"[bold]{label}[/]\n" + "\n".join(rows))
    di = next((k for k, v in mapping.items() if v == ("val", default)), 1)
    n = IntPrompt.ask("  #", default=di)
    kind, val = mapping.get(n, mapping[di])
    if kind == "none":
        return None
    if kind == "other":
        return Prompt.ask("  timm model id")
    if kind == "search":
        import timm
        q = Prompt.ask("  Search substring (e.g. vit, convnext, resnet)")
        matches = timm.list_models(f"*{q}*")
        if not matches:
            console.print(f"  [yellow]no matches for '{q}'[/] — type it directly")
            return Prompt.ask("  timm model id")
        console.print(f"  {len(matches)} match(es)" + (" — showing 30" if len(matches) > 30 else ""))
        return _pick("Matches", matches[:30], default=matches[0], allow_other=True)
    return val


def _list_configs() -> list[str]:
    d = ROOT / "configs"
    return sorted(f"configs/{p.name}" for p in d.glob("*.yaml")) if d.exists() else []


def _list_checkpoints() -> list[str]:
    d = ROOT / "checkpoints"
    return sorted(str(p.relative_to(ROOT)) for p in d.rglob("*.pt"))[:40] if d.exists() else []


def _confirm_run(cmd: list[str]) -> None:
    """Show a command and run it after confirmation, returning to the menu after."""
    from rich.prompt import Confirm
    console.print(Panel(" ".join(cmd), title="command", border_style="cyan", expand=False))
    if Confirm.ask("Run it now?", default=False):
        subprocess.run(cmd, cwd=str(ROOT))
    else:
        console.print("[yellow]Not executed.[/] Copy the command above to run it elsewhere "
                      "(e.g. Verode/Kaggle).")


@app.command()
def menu() -> None:
    """Interactive guided menu — pick an action and fill in the parameters.

    Each action asks the common parameters and then offers an "Advanced options"
    gate that exposes every flag the equivalent command accepts, so nothing is
    reachable only by typing the raw command."""
    from rich.prompt import Prompt, IntPrompt, Confirm

    def _opt(q, default=""):
        """Optional free-text prompt → value or None."""
        return Prompt.ask(q, default=default) or None

    def _opt_int(q):
        v = Prompt.ask(q, default="")
        return int(v) if v.strip() else None

    from src.performance_model import MODEL_TABLE, GPU_TABLE
    models_l = list(MODEL_TABLE)
    gpus_l = list(GPU_TABLE)

    while True:
        console.print(Panel(
            "[bold]1[/] · Predict (no GPU)\n[bold]2[/] · Train\n"
            "[bold]3[/] · Feasibility benchmark\n[bold]4[/] · Evaluate on test\n"
            "[bold]5[/] · Open dashboard\n[bold]0[/] · Exit",
            title="TFG — what do you want to do?", border_style="cyan", expand=False))
        choice = Prompt.ask("Choose", choices=["0", "1", "2", "3", "4", "5"], default="1")

        if choice == "0":
            return
        if choice == "1":
            strat = Prompt.ask("Strategy", choices=["single", "ddp", "model_parallel",
                               "heterogeneous"], default="single")
            _show_prediction(
                model=_pick("Model", models_l, default="vit_base_patch16_224"),
                gpu=_pick("GPU", gpus_l, default="Tesla T4"),
                strategy=strat,
                n_gpus=IntPrompt.ask("GPUs", default=1 if strat == "single" else 2),
                batch=IntPrompt.ask("Global batch", default=96),
                precision=Prompt.ask("Precision", choices=["fp32", "amp", "tf32", "bf16"], default="fp32"),
                dataset_size=resolve_dataset_n(
                    Prompt.ask("Dataset", choices=["subset", "full"], default="subset")),
                epochs=IntPrompt.ask("Epochs", default=15),
                disk=Prompt.ask("Disk", choices=["ssd", "nvme", "hdd", "nfs"], default="ssd"))
        elif choice == "2":
            strat = Prompt.ask("Strategy", choices=list(STRATEGIES), default="single")
            config = _pick("Config", _list_configs(), default="configs/train.yaml")
            kw = dict(model=_pick_model("Model", allow_from_config=True),
                      epochs=_opt_int("Epochs (blank = from config)"),
                      trace=Prompt.ask("Trace", choices=["off", "simple", "deep"], default="simple"))
            if strat in ("ddp", "heterogeneous"):
                kw["n_gpus"] = IntPrompt.ask("GPUs per node", default=2)
            if Confirm.ask("Advanced options (precision, layers, fn, multi-node)?", default=False):
                if strat == "single":
                    kw["precision"] = Prompt.ask("Precision",
                                                 choices=["fp32", "amp", "tf32", "bf16"], default="fp32")
                kw["layers"] = _csv(_opt("Layers (comma: plot,confusion,batch-monitor,hooks)"))
                kw["fn"] = _csv(_opt("Fn decorators (comma: timing,energy)"))
                if strat in ("ddp", "heterogeneous"):
                    nnodes = IntPrompt.ask("Number of nodes", default=1)
                    kw["nnodes"] = nnodes
                    if nnodes > 1:
                        kw["node_rank"] = IntPrompt.ask("This node's rank", default=0)
                        kw["master_addr"] = Prompt.ask("Master address", default="localhost")
                        kw["master_port"] = IntPrompt.ask("Master port", default=29500)
            _confirm_run(build_train_cmd(strat, config, **kw))
        elif choice == "3":
            models = [_pick_model("Model")]
            bsizes = [int(x) for x in (_csv(Prompt.ask("Batch sizes (comma)", default="32,64")) or [])]
            ep = IntPrompt.ask("Epochs (for the time estimate)", default=30)
            traces = _csv(Prompt.ask("Trace modes (comma: off,simple,deep)", default="off,simple"))
            kw = {}
            if Confirm.ask("Advanced options (config, dataset path, precision, study, NFS, device)?",
                           default=False):
                kw["config"] = _pick("YAML config", _list_configs(), none_label="(none)")
                kw["dataset_path"] = _opt("Dataset path to measure real I/O (blank = none)")
                kw["precision"] = Prompt.ask("Precision", choices=["fp32", "amp", "tf32", "bf16"],
                                             default="fp32")
                kw["compare_precision"] = Confirm.ask("Compare FP32 vs Tensor cores?", default=False)
                kw["convergence_study"] = Confirm.ask("Real convergence study (mini-training)?",
                                                      default=False)
                if kw["convergence_study"]:
                    kw["study_steps"] = IntPrompt.ask("Mini-training steps", default=60)
                kw["nfs_factor"] = float(Prompt.ask("NFS factor (Verode ≈ 1.3)", default="1.0"))
                kw["device"] = IntPrompt.ask("CUDA device index", default=0)
            _confirm_run(build_feasibility_cmd(models, bsizes or None, ep, traces, **kw))
        elif choice == "4":
            _ckpts = _list_checkpoints()
            ckpt = (_pick("Checkpoint", _ckpts, allow_other=True) if _ckpts
                    else Prompt.ask("Checkpoint path"))
            config = _pick("Config", _list_configs(), default="configs/train.yaml")
            split = Prompt.ask("Split", choices=["test", "val"], default="test")
            kw = {}
            if Confirm.ask("Advanced options (model, output CSV, metadata, batch size, max batches)?",
                           default=False):
                kw["model"] = _pick_model("Model", allow_from_config=True)
                kw["output"] = _opt("Output CSV (blank = none)")
                kw["metadata"] = _opt("Metadata parquet (blank = from config)")
                kw["batch_size"] = _opt_int("Batch size (blank = from config)")
                kw["max_batches"] = _opt_int("Max batches (blank = all)")
            _confirm_run(build_eval_cmd(ckpt, config, split=split, **kw))
        elif choice == "5":
            port = IntPrompt.ask("Port", default=8501)
            _confirm_run([sys.executable, "-m", "streamlit", "run", "src/web/app.py",
                          "--server.port", str(port)])
        console.print()   # blank line before the menu shows again


if __name__ == "__main__":
    app()
