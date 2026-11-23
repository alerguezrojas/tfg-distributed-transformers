"""Choose an ensemble of runs on validation and evaluate it once on test.

Each run folder holds ``val_preds.npz`` and ``test_preds.npz`` (``paravit eval --split val|test
--save-predictions``). The selection (``greedy`` of Caruana et al. 2004, or ``per-group``: the best
run of each group, and the number of groups k of best validation) uses **only validation**; the
chosen set is then evaluated a single time on test, optionally against a reference ensemble with
the paired bootstrap. No GPU, no torch: pure NumPy.

Usage:
    uv run python scripts/select_ensemble.py \\
        --runs logs/.../vit_s42 logs/.../swin_s42 logs/.../deit_s42 \\
        --group-by groups.csv --method per-group \\
        --reference logs/.../vit_s42 logs/.../vit_s43 logs/.../vit_s44 --markdown
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.compare import (METRICS, check_aligned, default_metrics,  # noqa: E402
                                  default_primary, paired_bootstrap, run_metrics)
from src.analysis.ensemble import (average, best_per_group, choose_k,  # noqa: E402
                                   greedy_selection, topk_curve)
from src.analysis.io import load_predictions  # noqa: E402
from compare_predictions import _ci, _fmt, _table  # noqa: E402


def run_names(dirs: list[str]) -> list[str]:
    """One distinct name per run folder: its folder name, numbered when it repeats."""
    names = [Path(d).name for d in dirs]
    if len(set(names)) != len(names):
        names = [f"{n} #{i + 1}" for i, n in enumerate(names)]
    return names


def read_group_map(path: str) -> dict[str, str]:
    """``run name -> group`` from a CSV (``run,group`` per line, optional header) or YAML mapping.

    The file must live under the working directory (real path, so ``..`` or a symlink cannot
    point elsewhere): it is a small table that goes with the experiment, not an arbitrary path."""
    base = os.path.realpath(os.getcwd())
    target = os.path.realpath(path)
    if os.path.commonpath([base, target]) != base:
        raise SystemExit(f"--group-by: {path} must be inside the working directory ({base})")
    p = Path(target)
    if not p.is_file():
        raise SystemExit(f"--group-by: file not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"--group-by: {path} must be a mapping run: group")
        return {str(k): str(v) for k, v in data.items()}
    out = {}
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() and not row[0].lstrip().startswith("#"):
                out[row[0].strip()] = row[1].strip()
    if out.get("run") in ("group", "grupo"):        # header row
        del out["run"]
    return out


def _load(dirs: list[str], fname: str, what: str):
    preds = []
    for d in dirs:
        f = Path(d) / fname
        if not f.is_file():
            raise SystemExit(f"Missing {what} predictions: {f} (run `paravit eval --split "
                             f"{what} --save-predictions` on that run)")
        q = load_predictions(f)
        if q.task != "multiclass":
            raise SystemExit(f"Only multiclass predictions are supported (got task={q.task!r} in {f}).")
        preds.append(q)
    return preds


def _check(preds: list, names: list[str], what: str) -> None:
    if len({q.ordinal for q in preds}) != 1 or any(q.classes != preds[0].classes for q in preds):
        raise SystemExit(f"All {what} predictions must share the same classes and ordinal flag.")
    try:
        check_aligned([q.targets for q in preds])
        for key in sorted(set.intersection(*(set(q.groups) for q in preds))):
            check_aligned([q.targets for q in preds], [q.groups[key] for q in preds])
    except ValueError as e:
        raise SystemExit(f"Error in the {what} predictions of {names}: {e}")


def select(names, val_probs, targets, score_fn, method, group_of, max_size) -> dict:
    """Run the selection on validation; returns ``chosen`` (names, repeated = weight) and ``trace``
    rows ``(step label, detail, validation score)``."""
    if method == "greedy":
        chosen, trace = greedy_selection(val_probs, targets, score_fn, max_size=max_size)
        return {"chosen": chosen, "trace": [(f"+ {n}", "", s) for n, s in trace]}
    scores = {n: score_fn(val_probs[n], targets) for n in names}
    ranked = best_per_group(scores, group_of)
    order = [r for _, r, _ in ranked]
    curve = topk_curve(val_probs, targets, order, score_fn)
    k = choose_k(curve)
    trace = [(f"k = {i}", f"{ranked[i - 1][0]}: {ranked[i - 1][1]}", s) for i, s in curve]
    return {"chosen": order[:k], "trace": trace, "k": k}


def render(info: dict, markdown: bool = False) -> str:
    """Text or Markdown report of :func:`build_report`."""
    c, prim, ms = markdown, info["primary"], info["metrics"]
    lab = {m: METRICS[m][1] for m in ms}
    out = [f"Selección de conjunto ({info['method']}) sobre {info['n_val']} muestras de validación; "
           f"evaluación única sobre {info['n_test']} de test. Métrica de selección: {lab[prim]}.",
           "La selección usa solo validación: el test no interviene en ninguna decisión.", ""]
    rows = [[n, _fmt(v, 3, comma=c), _fmt(t, 3, comma=c), g]
            for n, v, t, g in info["candidates"]]
    out += [_table(["Run", f"{lab[prim]} val", f"{lab[prim]} test", "Grupo"], rows, c), ""]
    out.append("Traza de la selección (solo validación):")
    out.append("")
    out += [_table(["Paso", "Detalle", f"{lab[prim]} val"],
                   [[a, b, _fmt(s, 3, comma=c)] for a, b, s in info["trace"]], c), ""]
    weights = {n: info["chosen"].count(n) for n in dict.fromkeys(info["chosen"])}
    out.append("Conjunto elegido: " + ", ".join(n + (f" ×{w}" if w > 1 else "")
                                                for n, w in weights.items()) + ".")
    out += ["", f"Evaluación única en test del conjunto elegido ({sum(weights.values())} miembros):", ""]
    rows = [[lab[m], _fmt(info["test"][m], 3, comma=c)] for m in ms]
    out += [_table(["Métrica", "Test"], rows, c)]
    if info.get("diff"):
        out += ["", f"Bootstrap emparejado en test ({info['n_boot']} remuestreos, semilla {info['seed']}), "
                f"elegido − referencia ({', '.join(info['reference'])}), IC 95 % por percentiles:", ""]
        rows = [[lab[m], _fmt(info["ref"][m], 3, comma=c), _ci(info["diff"][m], c),
                 _fmt(info["diff"][m]["p"], 3, comma=c)] for m in ms]
        out.append(_table(["Métrica", "Referencia", "Diferencia [IC 95 %]", "P(d ≤ 0)"], rows, c))
    return "\n".join(out)


def build_report(args) -> dict:
    names = run_names(args.runs)
    val = _load(args.runs, args.val_name, "val")
    test = _load(args.runs, args.test_name, "test")
    _check(val, names, "val")
    _check(test, names, "test")
    if test[0].classes != val[0].classes or test[0].ordinal != val[0].ordinal:
        raise SystemExit("Validation and test predictions must share the same classes and ordinal flag.")
    ordinal = val[0].ordinal
    ref_test = ref_names = None
    if args.reference:
        ref_names = run_names(args.reference)
        ref_test = _load(args.reference, args.test_name, "test")
        _check(test + ref_test, names + ref_names, "test")
    group_of = {n: n for n in names}
    if args.group_by:
        gm = read_group_map(args.group_by)
        unknown = sorted(set(gm) - set(names))
        if unknown:
            raise SystemExit(f"--group-by names unknown runs: {unknown}; runs are {names}")
        group_of = {n: gm.get(n, n) for n in names}     # a run without a row is its own group
    prim, metrics = default_primary(ordinal), default_metrics(ordinal)
    score_fn = METRICS[prim][0]
    vp = {n: q.probs.astype(np.float64) for n, q in zip(names, val)}
    tp = {n: q.probs.astype(np.float64) for n, q in zip(names, test)}
    try:
        sel = select(names, vp, val[0].targets, score_fn, args.method, group_of, args.max_size)
        t_targets = test[0].targets
        ens = average([tp[n] for n in sel["chosen"]])
        info = {"method": args.method, "primary": prim, "metrics": metrics, "chosen": sel["chosen"],
                "trace": sel["trace"], "n_val": len(val[0].targets), "n_test": len(t_targets),
                "candidates": [(n, score_fn(vp[n], val[0].targets), score_fn(tp[n], t_targets),
                                group_of[n]) for n in names],
                "test": run_metrics(ens, t_targets, metrics=metrics, primary=prim, ordinal=ordinal)}
        if ref_test:
            ref = average([q.probs for q in ref_test])
            idx = np.random.default_rng(args.seed).integers(0, len(t_targets),
                                                            size=(args.n_boot, len(t_targets)))
            info["diff"] = {m: paired_bootstrap(
                lambda i, m=m: METRICS[m][0](ens[i], t_targets[i]),
                lambda i, m=m: METRICS[m][0](ref[i], t_targets[i]),
                len(t_targets), indices=idx) for m in metrics}
            info["ref"] = run_metrics(ref, t_targets, metrics=metrics, primary=prim, ordinal=ordinal)
            info.update(reference=ref_names, n_boot=args.n_boot, seed=args.seed)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", action="extend", required=True, metavar="DIR",
                   help="Run folders, each with its val and test predictions.")
    p.add_argument("--val-name", default="val_preds.npz")
    p.add_argument("--test-name", default="test_preds.npz")
    p.add_argument("--group-by", default=None, metavar="FILE",
                   help="CSV (run,group) or YAML (run: group); default: every run is its own group.")
    p.add_argument("--method", choices=("greedy", "per-group"), default="per-group")
    p.add_argument("--max-size", type=int, default=8, help="Greedy: maximum ensemble size.")
    p.add_argument("--reference", nargs="+", action="extend", default=None, metavar="DIR",
                   help="Run folders of the reference ensemble (paired bootstrap on test).")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markdown", action="store_true",
                   help="Markdown tables with decimal comma, ready for docs/datasets/*.md.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    print(render(build_report(args), markdown=args.markdown))


if __name__ == "__main__":
    main()
