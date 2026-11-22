"""Paired comparison of two groups of runs from their saved test predictions.

Reads the ``.npz`` of ``scripts/eval.py --save-predictions`` (``probs``, ``targets`` and,
optionally, ``group__<name>``) and reports, for group A against group B (e.g. three seeds of a
variant against three of the base): the metrics of every run, mean ± std and the ensemble of each
group, and the paired bootstrap of the difference (mean of the seeds, ensembles and pairs of
seeds). No GPU, no torch: pure NumPy.

Usage:
    uv run python scripts/compare_predictions.py \\
        --a runs/clean_s42/test_preds.npz runs/clean_s43/test_preds.npz runs/clean_s44/test_preds.npz \\
        --b runs/base_s42/test_preds.npz runs/base_s43/test_preds.npz runs/base_s44/test_preds.npz \\
        --label-a Limpio --label-b Base --group data_provider --markdown
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.compare import (METRICS, check_aligned, compare_groups,  # noqa: E402
                                  default_metrics, default_primary)
from src.analysis.io import load_predictions  # noqa: E402


def load_group(paths: list[str]):
    """Load the files of a group: ``(Predictions list, names)``; names from the file or its folder."""
    preds = [load_predictions(p) for p in paths]
    stems = [Path(p).stem for p in paths]
    names = stems if len(set(stems)) == len(stems) else [Path(p).parent.name for p in paths]
    if len(set(names)) != len(names):
        names = [f"{n} #{i + 1}" for i, n in enumerate(names)]
    return preds, names


def _fmt(x: float, nd: int = 3, sign: bool = False, comma: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    s = s.replace("-", "−")          # real minus sign, as in docs/datasets/*.md
    return s.replace(".", ",") if comma else s


def _ci(r: dict, comma: bool) -> str:
    return f"{_fmt(r['diff'], 3, True, comma)} [{_fmt(r['lo'], 3, True, comma)}; {_fmt(r['hi'], 3, True, comma)}]"


def _table(header: list[str], rows: list[list[str]], markdown: bool) -> str:
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])
    w = [max(len(c) for c in col) for col in zip(header, *rows)]
    fmt = lambda r: "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
    return "\n".join([fmt(header), "  ".join("-" * x for x in w)] + [fmt(r) for r in rows])


def render(res: dict, names_a: list[str], names_b: list[str], label_a: str, label_b: str,
           markdown: bool = False) -> str:
    """Tables of the result of :func:`compare_groups` (Markdown with decimal comma, or plain text)."""
    c = markdown
    ms, prim = res["metrics"], res["primary"]
    lab = {m: METRICS[m][1] for m in ms}
    groups = list(res["summary"]["a"]["groups"])
    head = ["Run", *[lab[m] for m in ms], *[f"{lab[prim]} {g}" for g in groups]]

    def run_rows(key, names, label):
        return [[f"{label}: {n}", *[_fmt(r[m], 3, comma=c) for m in ms],
                 *[_fmt(r["groups"][g], 3, comma=c) for g in groups]]
                for n, r in zip(names, res["runs"][key])]

    def ms_cell(v):
        return f"{_fmt(v[0], 3, comma=c)} ± {_fmt(v[1], 3, comma=c)}"

    out = [f"Comparación de {label_a} (n = {len(names_a)}) con {label_b} (n = {len(names_b)}); "
           f"{res['n']} muestras de test, métrica principal: {lab[prim]}.", "",
           _table(head, run_rows("a", names_a, label_a) + run_rows("b", names_b, label_b), c), ""]
    rows = []
    for key, label in (("a", label_a), ("b", label_b)):
        s = res["summary"][key]
        e = s["ensemble"]
        rows.append([label, *[ms_cell(s[m]) for m in ms], *[ms_cell(s["groups"][g]) for g in groups],
                     _fmt(e[prim], 3, comma=c)])
    out += [_table(["Grupo", *head[1:], f"Conjunto ({lab[prim]})"], rows, c), ""]
    out.append(f"Bootstrap emparejado ({res['n_boot']} remuestreos, semilla {res['seed']}, mismos "
               f"índices para todos los runs), {label_a} − {label_b}, IC 95 % por percentiles:")
    out.append("")
    rows = []
    for m in ms:
        r = res["diff"]["mean"][m]
        rows.append(["Media de las semillas", lab[m], _ci(r, c), _fmt(r["p"], 3, comma=c)])
    for m in ms:
        r = res["diff"]["ensemble"][m]
        rows.append(["Conjunto", lab[m], _ci(r, c), _fmt(r["p"], 3, comma=c)])
    for i, pair in enumerate(res["diff"]["pairs"]):
        for m in ms:
            r = pair[m]
            rows.append([f"Par {i + 1} ({names_a[i]} − {names_b[i]})", lab[m], _ci(r, c),
                         _fmt(r["p"], 3, comma=c)])
    out.append(_table(["Comparación", "Métrica", "Diferencia [IC 95 %]", "P(d ≤ 0)"], rows, c))
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--a", nargs="+", action="extend", required=True, metavar="NPZ",
                   help="Predictions of group A (one .npz per run).")
    p.add_argument("--b", nargs="+", action="extend", required=True, metavar="NPZ",
                   help="Predictions of group B (the reference).")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--group", default=None,
                   help="Subgroup variable (key group__<name> of the .npz), e.g. data_provider.")
    p.add_argument("--metrics", default=None,
                   help=f"Comma-separated subset of {list(METRICS)} (default: all that apply).")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markdown", action="store_true",
                   help="Markdown tables with decimal comma, ready for docs/datasets/*.md.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    pa, na = load_group(args.a)
    pb, nb = load_group(args.b)
    allp = pa + pb
    for q in allp:
        if q.task != "multiclass":
            raise SystemExit(f"Only multiclass predictions are supported (got task={q.task!r}).")
    if len({q.ordinal for q in allp}) != 1 or any(q.classes != allp[0].classes for q in allp):
        raise SystemExit("All runs must share the same classes and ordinal flag.")
    ordinal = allp[0].ordinal
    groups = None
    if args.group:
        missing = [n for q, n in zip(allp, na + nb) if args.group not in q.groups]
        if missing:
            raise SystemExit(f"--group {args.group!r} is missing in: {missing}")
        groups = [q.groups[args.group] for q in allp]
    try:
        check_aligned([q.targets for q in allp], groups)
        metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else default_metrics(ordinal)
        res = compare_groups([q.probs for q in pa], [q.probs for q in pb], pa[0].targets,
                             groups[0] if groups else None, metrics=metrics,
                             primary=default_primary(ordinal), ordinal=ordinal,
                             n_boot=args.n_boot, seed=args.seed)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")
    print(render(res, na, nb, args.label_a, args.label_b, markdown=args.markdown))


if __name__ == "__main__":
    main()
