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
cost of the architectures). The last section checks the integrated predictor
(``src/performance_model.predict``, three terms in mixed precision) against the epoch time of
real PANDA trainings, against its former FLOPs-only form. Results in docs/performance_model.md.

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
    fit_with_overhead,
    fit_two_term,
    leave_one_out,
    predict_ips,
    relative_errors,
    shape_errors,
    transfer,
)

DATA = ROOT / "docs" / "research" / "datos" / "throughput_arquitecturas_amp.csv"


def _mem_bw() -> dict[str, float]:
    """Datasheet memory bandwidth (GB/s) of every GPU in the predictor's GPU_TABLE."""
    from src.performance_model import GPU_TABLE
    return {name: g.mem_bw_gbs for name, g in GPU_TABLE.items() if g.mem_bw_gbs > 0}


# Real PANDA trainings (512 px, AMP, 7,433 train / 1,593 val images) whose epoch time checks
# the integrated predictor: (strategy, GPU, n GPUs, global batch, logs folder, models→runs).
# Real = mean over the runs of the median epoch_time_s without the first epoch (train + val).
TRAININGS = (
    ("single", "RTX 3060 Ti", 1, 16, "logs/local/panda/single", {
        "vit_small_patch16_224": ("21112026_061527", "21112026_064000", "21112026_070409"),
        "deit3_small_patch16_224.fb_in22k_ft_in1k": ("22112026_154858",),
        "eva02_small_patch14_224.mim_in22k": ("22112026_174205",),
        "swin_tiny_patch4_window7_224": ("22112026_160932", "22112026_182151", "22112026_190007"),
        "convnext_tiny.fb_in22k": ("22112026_152254",),
        "resnet50": ("22112026_164749",),
        "efficientnet_b0": ("22112026_150517",),
        "tf_efficientnetv2_s": ("22112026_171208",),
    }),
    ("ddp", "Tesla T4", 2, 16, "logs/kaggle/panda/ddp", {
        "vit_small_patch16_224": ("20112026_163243",),
        "vit_base_patch16_224.augreg2_in21k_ft_in1k": ("22112026_201056", "23112026_104927",
                                                       "23112026_125243"),
        "swin_tiny_patch4_window7_224": ("23112026_135517",),
        "swinv2_tiny_window8_256": ("22112026_220303", "23112026_095246", "23112026_115634"),
        "convnext_small.fb_in22k": ("22112026_210603",),
        "maxvit_tiny_tf_512.in1k": ("22112026_234538",),
        "efficientnet_b3": ("22112026_225945",),
    }),
)
PANDA_TRAIN, PANDA_VAL = 7433, 1593


def load(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    """``{(gpu, img_size): {model: train imgs/s}}`` from the tidy CSV (always the versioned
    one: the study reads no path from the command line). All rows use batch 16."""
    out: dict[tuple[str, int], dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault((row["gpu"], int(row["img_size"])), {})[row["model"]] = float(row["train_ips"])
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


BATCH = 16      # every measurement of the CSV uses batch 16


def resolution_section(data, feats_at, cal_gpu: str, cal_size: int, md: bool) -> None:
    """Same GPU, other resolutions: the fits of ``cal_size`` predict the rest, without
    refitting, with two terms and with the fixed cost per step (``d/B``)."""
    cal = data[(cal_gpu, cal_size)]
    names = list(cal)
    F = np.array([feats_at(cal_size)[m][0] for m in names])
    A = np.array([feats_at(cal_size)[m][1] for m in names])
    R = np.array([cal[m] for m in names])
    a1 = fit_flops_only(F, R)
    a2, b2 = fit_two_term(F, A, R)
    a3, b3, d3 = fit_with_overhead(F, A, R, np.full(len(R), BATCH))
    print(f"Otras resoluciones en {cal_gpu}, sin reajustar (calibrado solo a {cal_size} px; "
          f"coste fijo por paso d = {d3 * 1e3:.1f} ms):\n")
    mean = lambda e: f"{100 * np.mean(np.abs(e)):.0f} %"
    for (gpu, size), meas in sorted(data.items()):
        if gpu != cal_gpu or size == cal_size:
            continue
        ms = list(meas)
        Ft = np.array([feats_at(size)[m][0] for m in ms])
        At = np.array([feats_at(size)[m][1] for m in ms])
        Rt = np.array([meas[m] for m in ms])
        e1 = relative_errors(predict_ips(a1, 0, Ft, At), Rt)
        e2 = relative_errors(predict_ips(a2, b2, Ft, At), Rt)
        e3 = relative_errors(predict_ips(a3, b3, Ft, At, d3, BATCH), Rt)
        rows = [[m, f"{Rt[i]:.1f}", _pct(e1[i]), _pct(e2[i]), _pct(e3[i])] for i, m in enumerate(ms)]
        print(f"{size} px:\n")
        print(_table(["Modelo", "img/s", "Solo FLOPs", "2 términos", "2 términos + d/B"], rows, md))
        print(f"\n|error| medio: solo FLOPs {mean(e1)}, 2 términos {mean(e2)}, con coste fijo "
              f"{mean(e3)} (máx {100 * np.max(np.abs(e3)):.0f} %)\n")


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
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--calibration-gpu", default="RTX 3060 Ti")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args(argv)
    from src.performance_model import gpu_spec

    data = load(DATA)
    cal = data.get((args.calibration_gpu, args.img_size))
    if not cal:
        raise SystemExit(f"no measurements for {args.calibration_gpu!r} at {args.img_size} px: "
                         f"{sorted(data)}")
    cache: dict[int, dict[str, tuple[float, float]]] = {}

    def feats_at(size: int) -> dict[str, tuple[float, float]]:
        if size not in cache:
            cache[size] = features(sorted({m for (g, s), d in data.items() if s == size for m in d}), size)
        return cache[size]

    feats = feats_at(args.img_size)
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
    for (gpu, size), meas in data.items():
        if gpu == args.calibration_gpu or size != args.img_size:
            continue
        g = gpu_spec(gpu)
        bw = _mem_bw()
        if g is None or gpu not in bw or args.calibration_gpu not in bw:
            raise SystemExit(f"no TFLOPS/π or bandwidth (GPU_TABLE) for {gpu!r} "
                             f"or {args.calibration_gpu!r}; add them to compare")
        k_c = (g_cal.fp32_tflops * g_cal.tensor_speedup) / (g.fp32_tflops * g.tensor_speedup)
        k_m = bw[args.calibration_gpu] / bw[gpu]
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
    resolution_section(data, feats_at, args.calibration_gpu, args.img_size, md)
    trainings_section(md)


def _median_epoch_s(folder: str, model: str, ts: str) -> float:
    """Median epoch_time_s of a run, without the first epoch (warm-up, cudnn autotune)."""
    with open(ROOT / folder / model / f"epoch_metrics_{ts}.csv", newline="") as f:
        rows = list(csv.DictReader(f))[1:]
    return float(np.median([float(r["epoch_time_s"]) for r in rows]))


def trainings_section(md: bool) -> None:
    """The integrated predictor against the epoch time of real PANDA trainings: three terms
    (what ``predict`` does in mixed precision) and the former FLOPs-only form (the same spec
    without its saved-activation figure)."""
    import dataclasses

    from src.model_profile import resolve_model_spec
    from src.performance_model import predict

    for strategy, gpu, n_gpus, batch, folder, runs in TRAININGS:
        rows, e_old, e_new = [], [], []
        for model, stamps in runs.items():
            real = float(np.mean([_median_epoch_s(folder, model, ts) for ts in stamps]))
            spec = resolve_model_spec(model, 3, 6, 512)
            kw = dict(n_gpus=n_gpus, dataset_size=PANDA_TRAIN, val_size=PANDA_VAL, batch=batch,
                      precision="amp", epochs=1, disk_type="ssd")
            new = predict(strategy, model, gpu, model=spec, **kw)
            old = predict(strategy, model, gpu,
                          model=dataclasses.replace(spec, saved_gb_per_img=None), **kw)
            eo = old.time_per_epoch_total_s / real - 1
            en = new.time_per_epoch_total_s / real - 1
            e_old.append(eo)
            e_new.append(en)
            rows.append([model, str(len(stamps)), f"{real:.0f} s",
                         f"{old.time_per_epoch_total_s:.0f} s", _pct(eo),
                         f"{new.time_per_epoch_total_s:.0f} s", _pct(en)])
        print(f"Entrenamientos reales de PANDA: {strategy} en {n_gpus} × {gpu}, lote global "
              f"{batch}, AMP, 512 px (tiempo por época con validación):\n")
        print(_table(["Modelo", "Runs", "Real", "Solo FLOPs", "Error", "3 términos", "Error"],
                     rows, md))
        print(f"\n|error| medio: solo FLOPs {100 * np.mean(np.abs(e_old)):.0f} %, 3 términos "
              f"{100 * np.mean(np.abs(e_new)):.0f} % (máx. {100 * np.max(np.abs(e_new)):.0f} %)\n")


if __name__ == "__main__":
    main()
