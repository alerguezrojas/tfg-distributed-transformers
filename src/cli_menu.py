"""Interactive guided menu for the ``paravit`` CLI.

Separated from ``src/cli.py`` (which defines the commands) so each file has a single
responsibility: ``cli.py`` declares the Typer commands and their pure argv builders,
this module owns the interactive walkthrough. ``cli.menu()`` is a thin wrapper that
calls :func:`run_menu`; the import is lazy there, so there is no import cycle.
"""
from __future__ import annotations

import subprocess
import sys

from rich.panel import Panel

from src.cli import (
    console, ROOT, STRATEGIES, _csv, resolve_dataset_n, _show_prediction,
    build_train_cmd, build_benchmark_cmd, build_eval_cmd,
)


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


def run_menu() -> None:
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
            "[bold]1[/] · Estimate (analytic, no GPU)\n[bold]2[/] · Train\n"
            "[bold]3[/] · Benchmark (empirical, real)\n[bold]4[/] · Evaluate on test\n"
            "[bold]5[/] · Open dashboard\n[bold]0[/] · Exit",
            title="ParaViTLab — what do you want to do?", border_style="cyan", expand=False))
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
            _confirm_run(build_benchmark_cmd(models, bsizes or None, ep, traces, **kw))
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
