"""Predictor study: does training time follow FLOPs, or FLOPs + memory traffic?

Reads measured training throughput (``paravit benchmark``, synthetic batches) of many
architectures on several GPUs, profiles each model on CPU (forward GFLOPs and the activation
GB autograd saves per image, ``src/model_profile.py``) and compares two models of seconds
per image (``src/analysis/throughput_model.py``):

* FLOPs only, ``t = a·F`` — the form of the analytic predictor today;
* two terms, ``t = a·F + b·A`` — compute plus memory traffic.

On the calibration GPU it reports the in-sample fit and the leave-one-out error. On every
other GPU it predicts WITHOUT refitting, scaling ``a`` with (TFLOPS·π) and ``b`` with the
memory bandwidth of the spec sheets, and reports the absolute error, the common factor
(a GPU below its nominal clock) and the error once that factor is removed (the relative
cost of the architectures). Results in docs/performance_model.md.

    uv run python scripts/predictor_architecture_study.py            # data of 23/11/2026
    uv run python scripts/predictor_architecture_study.py --markdown
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.throughput_model import (  # noqa: E402
    common_factor,
    fit_flops_only,
    fit_two_term,
    leave_one_out,
    predict_ips,
    relative_errors,
    shape_errors,
    transfer,
)

DATA = ROOT / "docs" / "research" / "datos" / "throughput_arquitecturas_512_amp.csv"
# Memory bandwidth (GB/s) from the NVIDIA spec sheets; the predictor's GPU_TABLE has no
# bandwidth yet. TFLOPS and the Tensor-core factor π come from GPU_TABLE.
MEM_BW_GBS = {"RTX 3060 Ti": 448.0, "Tesla T4": 320.0, "Tesla V100": 900.0}


def load(path: Path) -> dict[str, dict[str, float]]:
    """``{gpu: {model: train imgs/s}}`` from the tidy CSV."""
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["gpu"], {})[row["model"]] = float(row["train_ips"])
    return out


def features(models, img_size: int, num_classes: int = 6) -> dict[str, tuple[float, float]]:
    """``{model: (forward GFLOPs, saved activation GB)}`` per image, profiled on CPU."""
    from src.model_profile import resolve_model_spec, saved_activation_gb
    feats = {}
    for m in models:
        spec = resolve_model_spec(m, 3, num_classes, img_size)
        if spec is None:
            raise SystemExit(f"cannot profile {m}")
        feats[m] = (spec.gflops_fwd, saved_activation_gb(m, 3, num_classes, img_size))
    return feats


def _pct(x: float) -> str:
    return f"{100 * x:+.0f} %"


def _table(header, rows, md: bool) -> str:
    if md:
        return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                         + ["| " + " | ".join(r) + " |" for r in rows])
    w = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--calibration-gpu", default="RTX 3060 Ti")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args(argv)
    from src.performance_model import gpu_spec

    data = load(Path(args.data))
    cal = data.get(args.calibration_gpu)
    if not cal:
        raise SystemExit(f"no measurements for {args.calibration_gpu!r}: {list(data)}")
    feats = features(sorted({m for d in data.values() for m in d}), args.img_size)
    names = list(cal)
    F = np.array([feats[m][0] for m in names])
    A = np.array([feats[m][1] for m in names])
    R = np.array([cal[m] for m in names])
    a, b = fit_two_term(F, A, R)
    a1 = fit_flops_only(F, R)
    e2, e1 = relative_errors(predict_ips(a, b, F, A), R), relative_errors(predict_ips(a1, 0, F, A), R)
    l2, l1 = leave_one_out(F, A, R), leave_one_out(F, A, R, two_term=False)
    md = args.markdown
    print(f"Calibración en {args.calibration_gpu} ({len(names)} modelos, {args.img_size} px): "
          f"a = {a:.3e} s/GFLOP, b = {b:.3e} s/GB (solo FLOPs: a = {a1:.3e})\n")
    rows = [[m, f"{F[i]:.1f}", f"{A[i]:.3f}", f"{F[i] / A[i]:.0f}", f"{R[i]:.1f}",
             _pct(e1[i]), _pct(l1[i]), _pct(e2[i]), _pct(l2[i])] for i, m in enumerate(names)]
    print(_table(["Modelo", "GFLOPs", "Activ. GB", "GFLOPs/GB", "img/s", "Solo FLOPs",
                  "Solo FLOPs (LOO)", "2 términos", "2 términos (LOO)"], rows, md))
    mean = lambda e: f"{100 * np.mean(np.abs(e)):.0f} %"
    print(f"\n|error| medio: solo FLOPs {mean(e1)} (LOO {mean(l1)}); "
          f"2 términos {mean(e2)} (LOO {mean(l2)}, máx {100 * np.max(np.abs(l2)):.0f} %)\n")
    g_cal = gpu_spec(args.calibration_gpu)
    for gpu, meas in data.items():
        if gpu == args.calibration_gpu:
            continue
        g = gpu_spec(gpu)
        if g is None or gpu not in MEM_BW_GBS or args.calibration_gpu not in MEM_BW_GBS:
            raise SystemExit(f"no TFLOPS/π (GPU_TABLE) or bandwidth (MEM_BW_GBS) for {gpu!r} "
                             f"or {args.calibration_gpu!r}; add them to compare")
        k_c = (g_cal.fp32_tflops * g_cal.tensor_speedup) / (g.fp32_tflops * g.tensor_speedup)
        k_m = MEM_BW_GBS[args.calibration_gpu] / MEM_BW_GBS[gpu]
        at, bt = transfer(a, b, k_c, k_m)
        ms = list(meas)
        Ft = np.array([feats[m][0] for m in ms])
        At = np.array([feats[m][1] for m in ms])
        Rt = np.array([meas[m] for m in ms])
        t2 = relative_errors(predict_ips(at, bt, Ft, At), Rt)
        t1 = relative_errors(predict_ips(a1 * k_c, 0, Ft, At), Rt)
        s2, s1 = shape_errors(t2), shape_errors(t1)
        print(f"Validación en {gpu}, sin reajustar (a ×{k_c:.2f} por TFLOPS·π, b ×{k_m:.2f} por "
              f"ancho de banda):\n")
        rows = [[m, f"{Rt[i]:.1f}", _pct(t1[i]), _pct(s1[i]), _pct(t2[i]), _pct(s2[i]),
                 "sí" if m in cal else "no"] for i, m in enumerate(ms)]
        print(_table(["Modelo", "img/s", "Solo FLOPs", "Solo FLOPs, forma", "2 términos",
                      "2 términos, forma", "En la calibración"], rows, md))
        ad, bd = fit_two_term(Ft, At, Rt)        # direct fit on this GPU, for diagnosis only
        print(f"\nAjuste directo en {gpu} (solo diagnóstico): a ×{ad / at:.2f} y b ×{bd / bt:.2f} "
              f"respecto a lo transferido por hoja técnica.")
        new = np.array([m not in cal for m in ms])
        print(f"\n|error| medio absoluto: solo FLOPs {mean(t1)}, 2 términos {mean(t2)}; factor común "
              f"(2 términos) {common_factor(t2):.2f}; |error| de forma: solo FLOPs {mean(s1)}, "
              f"2 términos {mean(s2)} (calibrados {mean(s2[~new])}, no calibrados "
              f"{mean(s2[new]) if new.any() else '—'})\n")


if __name__ == "__main__":
    main()
