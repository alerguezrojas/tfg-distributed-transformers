"""Benchmark — validate (predicted vs actual, Compare-style)."""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.benchmark_comparison import build_comparison
from src.web.benchmark_parser import (parse_ddp_scenarios, parse_benchmark_csv)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.helpers import (_get_benchmark_csvs, _get_runs, _load_df, _run_config,
                                _safe_max)


def _short(lbl: str) -> str:
    return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)


def _run_batch(r) -> int | None:
    """Per-GPU batch the run used (from its config line) — to match the benchmark
    row of the SAME batch (the benchmark is single-GPU, at that per-GPU batch)."""
    m = re.search(r"\d+", str(_run_config(str(r.log_path)).get("batch", "")))
    return int(m.group()) if m else None


def _run_ngpus(r) -> int:
    """Number of GPUs a distributed run used: global ÷ per-GPU batch if recorded,
    else 2 (the default for the project's DDP runs)."""
    txt = str(_run_config(str(r.log_path)).get("batch", ""))
    per = re.search(r"\d+", txt)
    glob = re.search(r"global\s*=?\s*(\d+)", txt)
    if per and glob and int(per.group()) > 0:
        return max(1, round(int(glob.group(1)) / int(per.group())))
    return 2


def _predicted_speedup(meta, n_gpus) -> float | None:
    """Predicted speedup at n_gpus from the benchmark report's DDP scenarios."""
    scen = parse_ddp_scenarios(meta)
    if not scen.empty and {"n_gpus", "speedup"}.issubset(scen.columns):
        row = scen[scen["n_gpus"] == n_gpus]
        if not row.empty:
            return float(row.iloc[0]["speedup"])
    return None


def _real_min_per_epoch(r) -> float | None:
    df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        return float(df["epoch_time"].mean()) / 60.0
    return None


# Default GPU per environment, used for the analytic prediction when the run has no
# matching benchmark report to read hardware_name from.
_ENV_GPU = {"kaggle": "Tesla T4", "verode": "Tesla V100", "local": "RTX 3060 Ti"}


def _run_gpu(r, meta) -> str | None:
    if meta and meta.get("hardware_name"):
        return str(meta.get("hardware_name"))
    return _ENV_GPU.get(r.env)


def _perf_strategy(mode: str) -> str:
    """Map a run's mode to a performance_model strategy name. The registry calls the
    GPU+CPU run 'ddp_hetero'; the analytic engine knows it as 'heterogeneous'."""
    return "heterogeneous" if mode == "ddp_hetero" else mode


def _run_sizes(r, meta) -> tuple[int | None, int | None]:
    """(n_train, n_val) for the run: from its config line, else the benchmark meta.
    Returns (None, None) when the size is genuinely unknown (a run with no config line
    matched to a benchmark with no #sizes) — we do NOT invent a default, because guessing
    the wrong size makes the estimate wildly off (a full-dataset run guessed as a 5 000
    subset, or vice-versa). n_val falls back to ~0.51× n_train (the BigEarthNet ratio)."""
    cfg = _run_config(str(r.log_path))

    def _i(v):
        m = re.search(r"\d+", str(v)) if v is not None else None
        return int(m.group()) if m else None

    n_train = _i(cfg.get("train")) or (_i(meta.get("n_train")) if meta else None)
    n_val = _i(cfg.get("val")) or (_i(meta.get("n_val")) if meta else None)
    if n_train is None:
        return None, None
    return n_train, (n_val or round(n_train * 0.51))


def _real_energy_per_epoch(df) -> float | None:
    """Mean measured energy per epoch (Wh): train (J→Wh) + eval (already Wh)."""
    total, have = 0.0, False
    if "energy_train_j" in df.columns and df["energy_train_j"].notna().any():
        total += float(df["energy_train_j"].mean()) / 3600.0
        have = True
    if "energy_eval_wh" in df.columns and df["energy_eval_wh"].notna().any():
        total += float(df["energy_eval_wh"].mean())
        have = True
    return total if have else None


def _cmp_metric(cmp, metric: str) -> tuple[float | None, float | None]:
    """(analytic, benchmark) for a metric from a BenchmarkComparison, or (None, None)."""
    if cmp is None:
        return None, None
    for row in cmp.rows:
        if row.metric == metric:
            return row.analytic, row.estimated
    return None, None


def _select_benchmark(parsed, env, model, bs):
    """Pick the best benchmark report for a run from `parsed` (list of
    (env, model, meta, df)). Same model is required; among those, prefer one that
    contains the run's batch size, then the same environment, then one that records its
    dataset size (#sizes). Returns (meta, df) or (None, None)."""
    cands = [(e, mo, m, df) for (e, mo, m, df) in parsed if mo == model]
    if not cands:
        return None, None

    def _score(item):
        e, _mo, m, df = item
        has_bs = bool(bs) and "batch_size" in getattr(df, "columns", []) and \
            (df["batch_size"] == bs).any()
        return (has_bs, e == env, m.get("n_train") is not None)

    e, _mo, m, df = max(cands, key=_score)
    return m, df


def _run_record(r, m, fdf) -> tuple[dict, object]:
    """The three-way (analytic / benchmark / real) record for one run, plus the
    BenchmarkComparison (or None). Pure given the run, its matched benchmark meta (m)
    and DataFrame (fdf); this is exactly what the charts and table render, so it is
    unit-testable and auditable without Streamlit.

    The three sources share the same physics (energy = power × time):
      - analytic : performance_model.predict() for the run's OWN config + dataset size,
      - benchmark: the empirical report's per-batch throughput, recomputed for the run's
                   size and corrected to its precision (÷ Tensor-core speedup) and GPU
                   count (time ÷ predicted DDP speedup),
      - real     : what the run measured.
    """
    run_df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
    bs = _run_batch(r)
    run_prec = (r.precision or "fp32")
    strat = _perf_strategy(r.mode)        # 'ddp_hetero' → 'heterogeneous' for the engine
    ng = _run_ngpus(r) if strat in ("ddp", "heterogeneous") else 1
    gpu = _run_gpu(r, m)
    n_train, n_val = _run_sizes(r, m)

    tensor_core = run_prec in ("amp", "fp16", "tf32", "bf16")
    _pc = (m.get("precision_cmp") or {}) if m else {}
    try:
        prec_sp = float(_pc.get("speedup")) if _pc.get("speedup") else None
    except (TypeError, ValueError):
        prec_sp = None
    precision_ok = (not tensor_core) or (prec_sp is not None)
    # Only DDP scales with the data-parallel speedup; heterogeneous is not modeled by the
    # benchmark's DDP scenarios (its analytic column handles that case instead).
    ddp_sp = _predicted_speedup(m, ng) if (m and r.mode == "ddp") else None

    # Real values, measured by the run (independent of the benchmark).
    real_t = _real_min_per_epoch(r)
    real_e = _real_energy_per_epoch(run_df)
    real_f1 = _safe_max(run_df["val_f1"]) if "val_f1" in run_df.columns else float("nan")
    real_f1 = None if (real_f1 is None or pd.isna(real_f1)) else float(real_f1)

    a_t = b_t = a_e = b_e = a_f = b_f = None
    note_bits: list[str] = []
    cmp = None
    if n_train is None:
        # No dataset size recorded (old run, no config line, benchmark with no #sizes):
        # we cannot estimate a per-epoch time/energy without it, so show only the real run.
        note_bits.append("dataset size not recorded — no estimate")
    if n_train is not None and m and fdf is not None and not fdf.empty and bs:
        nfs = float(m.get("nfs_factor", 1.0) or 1.0)
        cmp = build_comparison(
            meta=m, feas_df=fdf, actual_df=run_df, batch_size=bs, trace_mode="simple",
            nfs_factor=nfs, strategy=strat, gpu_name=gpu, n_gpus=ng, precision=run_prec,
            precision_speedup=(prec_sp if precision_ok else None), ddp_speedup=ddp_sp,
            run_n_train=n_train, run_n_val=n_val)
    if cmp is not None:
        a_t, b_t = _cmp_metric(cmp, "Total time / epoch")
        a_e, b_e = _cmp_metric(cmp, "Energy total / epoch")
        a_f, b_f = _cmp_metric(cmp, "Best Val F1")
        if strat in ("ddp", "heterogeneous") and ddp_sp:
            note_bits.append(f"bench ÷ {ddp_sp:.2f}× ({ng} GPU)")
        if tensor_core and prec_sp and precision_ok:
            note_bits.append(f"bench ÷ {prec_sp:.1f}× ({run_prec})")
    elif gpu and n_train is not None:
        # Pure-analytic fallback when no benchmark covers this batch/model (e.g. vit_large
        # MP @24, or a model with no report). Benchmark stays empty; analytic still applies.
        try:
            from src.performance_model import predict, expected_best_f1
            _gb = (bs or 96) * ng if strat in ("ddp", "heterogeneous") else (bs or 96)
            p = predict(strat, r.model, gpu, ng, dataset_size=n_train, batch=_gb,
                        precision=run_prec, epochs=1, val_size=n_val)
            if p:
                a_t = p.time_per_epoch_total_s / 60.0
                a_e = p.energy_per_epoch_wh
            a_f = expected_best_f1(r.model, n_train)[0]
            note_bits.append("analytic only (batch not benchmarked)")
        except Exception:
            pass

    # A precision the report never benchmarked → benchmark time/energy not comparable.
    if not precision_ok:
        b_t = b_e = None
        note_bits.append(f"≠ precision ({run_prec}, not benchmarked)")

    record = {
        "run": _short(r.label), "strategy": r.mode, "batch": bs, "precision": run_prec,
        "t": {"a": a_t, "b": b_t, "r": real_t},
        "e": {"a": a_e, "b": b_e, "r": real_e},
        "f": {"a": a_f, "b": b_f, "r": real_f1},
        "note": " · ".join(note_bits),
    }
    return record, cmp


# Fixed, easy-to-tell-apart colours for the three sources (red/green/blue).
_SRC_COLOR = {"analytic": "#2563eb", "benchmark": "#10b981", "real": "#ef4444"}


def _metric_block(records: list[dict], metric_key: str, title: str, ytitle: str,
                  key: str) -> None:
    """A per-metric block: a source picker (Analytic / Benchmark / Both — Real is always
    shown) followed by a grouped bar chart with the chosen sources, in red/green/blue."""
    recs = [{"run": d["run"], **{s: d[metric_key].get(s) for s in ("a", "b", "r")}}
            for d in records
            if any(d[metric_key].get(s) is not None for s in ("a", "b", "r"))]
    if not recs:
        st.caption(f"No data for **{title}** across the selected runs.")
        return
    choice = st.radio(f"{title} — compare Real with", ["Both", "Analytic", "Benchmark"],
                      horizontal=True, key=f"{key}_src")
    show = {"Both": ["a", "b"], "Analytic": ["a"], "Benchmark": ["b"]}[choice]
    labels = [d["run"] for d in recs]

    def _trace(src_key, name, color):
        ys = [d.get(src_key) for d in recs]
        return go.Bar(name=name, x=labels, y=ys, marker_color=color,
                      text=[f"{v:.2f}" if v is not None else "" for v in ys],
                      textposition="outside")

    fig = go.Figure()
    if "a" in show:
        fig.add_trace(_trace("a", "Analytic", _SRC_COLOR["analytic"]))
    if "b" in show:
        fig.add_trace(_trace("b", "Benchmark", _SRC_COLOR["benchmark"]))
    fig.add_trace(_trace("r", "Real", _SRC_COLOR["real"]))
    fig.update_layout(**_base_layout(360, title), barmode="group",
                      yaxis_title=ytitle, xaxis_title="")
    _show(fig, key)


def render_validate(ctx) -> object:
    """Predicted vs actual, Compare-style. Each single-GPU run is matched to the
    benchmark report of its model AND the batch size it actually used, so the
    estimate is for the same configuration the run ran (this is the fix for the
    'estimate very different from real' issue — the old code used the max-throughput
    batch, not the run's). Shows a table, a scorecard, a calibration scatter, the
    formula behind each estimate, and predicted-vs-real speedup."""
    st.markdown("### Estimate vs Benchmark vs Run")
    st.caption("Pick runs and compare the two **estimates** — the **analytic** prediction "
               "(formulas, no GPU) and the **empirical benchmark** — with what the runs "
               "**actually did**. Each run is matched to the benchmark report of its model "
               "and its batch size, computed for the run's own dataset size.")

    feas_csvs = _get_benchmark_csvs()
    if not feas_csvs:
        st.info("No benchmark reports yet. Generate one from the terminal "
                "(`paravit benchmark`).")
        st.divider()
        return st.container()

    parsed = []
    for p in feas_csvs:
        m, df = parse_benchmark_csv(p)
        env = p.parent.parent.name if p.parent.parent else "?"
        parsed.append((env, m.get("model_name", "?"), m, df))

    runs = _get_runs()
    labelled = {r.label: r for r in runs}
    feas_models = {mo for _, mo, _, _ in parsed}
    # Default prefers single-GPU runs (they get a full per-metric comparison) plus a
    # couple of distributed ones, so the table and the formula picker are populated.
    _sing = [r.label for r in runs if r.mode == "single" and r.model in feas_models]
    _oth = [r.label for r in runs if r.mode != "single" and r.model in feas_models]
    default = (_sing[:5] + _oth[:2])[:8] or list(labelled)[:3]
    sel = st.multiselect("Runs to compare against their estimate (max 8)",
                         list(labelled.keys()), default=default, max_selections=8)
    if not sel:
        st.info("Select at least one run.")
        st.divider()
        return st.container()

    # ── Per-run three-way extraction (analytic vs benchmark vs real run) ─────────
    # Each run is compared on three sources of the SAME physics:
    #   analytic  — closed-form performance_model.predict() (energy = power × time),
    #   benchmark — the empirical report, corrected to the run's precision/GPU count,
    #   real      — what the run actually measured.
    # Apples-to-apples needs the SAME precision: the benchmark is fp32, so for an AMP
    # run we divide its time/energy by the report's MEASURED Tensor-core speedup. If
    # the report never benchmarked that precision, the benchmark column is dropped.
    records = []
    cmp_by_run: dict = {}
    for lbl in sel:
        r = labelled[lbl]
        m, fdf = _select_benchmark(parsed, r.env, r.model, _run_batch(r))
        record, cmp = _run_record(r, m, fdf)
        records.append(record)
        if cmp is not None and r.mode == "single":
            cmp_by_run[lbl] = cmp      # the formula table is the single-GPU one

    # ── Summary table (analytic / benchmark / real, per metric) ──────────────────
    def _rnd(v, n=2):
        return round(v, n) if v is not None else None
    table_rows = [{
        "Run": d["run"], "Strategy": d["strategy"], "Batch": d["batch"], "Prec": d["precision"],
        "Time A": _rnd(d["t"]["a"]), "Time B": _rnd(d["t"]["b"]), "Time R": _rnd(d["t"]["r"]),
        "Wh A": _rnd(d["e"]["a"]), "Wh B": _rnd(d["e"]["b"]), "Wh R": _rnd(d["e"]["r"]),
        "F1 A": _rnd(d["f"]["a"], 3), "F1 B": _rnd(d["f"]["b"], 3), "F1 R": _rnd(d["f"]["r"], 3),
        "Note": d["note"],
    } for d in records]
    tdf = pd.DataFrame(table_rows).set_index("Run")
    st.dataframe(tdf, use_container_width=True,
                 column_config={"Note": st.column_config.TextColumn("Note", width="large")})
    _dl_csv(tdf.reset_index(), "three_way_comparison.csv", "Download comparison")
    st.caption("**A** = analytic prediction · **B** = empirical benchmark · **R** = real run. "
               "Time and Wh are per epoch (train+eval). The benchmark is fp32 single-GPU, so "
               "it is corrected to the run's precision (÷ measured Tensor-core speedup) and "
               "GPU count (time ÷ predicted DDP speedup; total energy conserved across GPUs "
               "only when DDP is compute-bound). Analytic and benchmark F1 are the same "
               "empirical prior — only **R** is a real measurement.")

    # ── Accuracy scorecard ──────────────────────────────────────────────────────
    def _mean_abs_err(key, src):
        errs = []
        for d in records:
            pred, real = d[key][src], d[key]["r"]
            if pred is not None and real not in (None, 0):
                errs.append(abs(pred - real) / abs(real) * 100)
        return (sum(errs) / len(errs)) if errs else None

    def _mean_f1_abs(src):
        errs = [abs(d["f"][src] - d["f"]["r"]) for d in records
                if d["f"][src] is not None and d["f"]["r"] is not None]
        return (sum(errs) / len(errs)) if errs else None

    def _pair(a, b):
        sa = f"A ±{a:.0f}%" if a is not None else "A —"
        sb = f"B ±{b:.0f}%" if b is not None else "B —"
        return f"{sa} · {sb}"

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Time error / epoch", _pair(_mean_abs_err("t", "a"), _mean_abs_err("t", "b")),
               help="Mean |predicted − real| / real of the per-epoch time, analytic (A) and "
                    "benchmark (B) vs the real run.")
    sc2.metric("Energy error / epoch", _pair(_mean_abs_err("e", "a"), _mean_abs_err("e", "b")),
               help="Mean |predicted − real| / real of the per-epoch energy.")
    _f1e = _mean_f1_abs("a")
    sc3.metric("Best Val F1 error", f"±{_f1e:.3f}" if _f1e is not None else "—",
               help="Mean |prior − real| best Val F1 (analytic and benchmark share the prior).")

    # ── Three grouped-bar charts: time, energy, F1 (analytic / benchmark / real) ─
    st.markdown("#### Per-metric comparison — analytic vs benchmark vs real")
    st.caption("Each chart shows the **Real** run (red) next to the source(s) you pick: "
               "**Analytic** (blue) and/or **Benchmark** (green).")
    _metric_block(records, "t", "Time per epoch (train+eval)", "Minutes", "validate_time_bars")
    _metric_block(records, "e", "Energy per epoch (train+eval)", "Wh", "validate_energy_bars")
    st.caption("Energy = effective power × time. Analytic power is calibrated per GPU on a "
               "compute-heavy workload (vit_base); AMP saves energy through time (same watts), "
               "so it tracks the time chart. It is most accurate for compute-bound runs and "
               "runs **high** when the GPU is under-used: light/I/O-bound models (vit_tiny "
               "draws ~47 W not ~64 W, and the demo subset is RAM-cached so the I/O floor "
               "over-counts time) and **heterogeneous** runs (the GPU idles near ~45 W waiting "
               "on the slow CPU worker, below its compute-bound power).")
    _metric_block(records, "f", "Best Val F1", "F1 (macro)", "validate_f1_bars")
    st.caption("Analytic and benchmark F1 are the SAME empirical prior (so they coincide "
               "for reports that record their dataset size; older reports may differ slightly); "
               "the real bar is the run's measured best Val F1.")

    # ── Calibration scatter (predicted vs real, with the diagonal) ──────────────
    metric = st.radio("Calibration plot", ["Time/epoch (min)", "Energy/epoch (Wh)", "Val F1"],
                      horizontal=True, key="cal_metric")
    _key = {"Time/epoch (min)": "t", "Energy/epoch (Wh)": "e", "Val F1": "f"}[metric]
    _unit = {"t": "min/epoch", "e": "Wh/epoch", "f": "Val F1"}[_key]
    pts = [d for d in records if d[_key]["r"] is not None
           and (d[_key]["a"] is not None or d[_key]["b"] is not None)]
    if pts:
        _vals = [v for d in pts for v in (d[_key]["a"], d[_key]["b"], d[_key]["r"]) if v is not None]
        _hi = (max(_vals) * 1.1) if _vals else 1.0
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=[0, _hi], y=[0, _hi], mode="lines",
                                  line=dict(color="#94a3b8", dash="dash"),
                                  name="perfect estimate", hoverinfo="skip"))
        for _src, _name, _col in (("a", "analytic", _SRC_COLOR["analytic"]),
                                  ("b", "benchmark", _SRC_COLOR["benchmark"])):
            _xs = [d[_key][_src] for d in pts if d[_key][_src] is not None]
            _ys = [d[_key]["r"] for d in pts if d[_key][_src] is not None]
            _tx = [d["run"] for d in pts if d[_key][_src] is not None]
            if _xs:
                figc.add_trace(go.Scatter(
                    x=_xs, y=_ys, mode="markers+text", text=_tx,
                    textposition="top center", textfont=dict(size=9),
                    marker=dict(size=12, color=_col, line=dict(width=1, color="white")),
                    name=_name,
                    hovertemplate="%{text}<br>predicted %{x:.2f}<br>real %{y:.2f}<extra></extra>"))
        figc.update_layout(**_base_layout(360, f"Predicted vs real — {_unit}"),
                           xaxis_title=f"Predicted ({_unit})", yaxis_title=f"Real ({_unit})")
        _show(figc, "validate_calibration")
        st.caption("On the dashed diagonal = perfect estimate. **Blue** = analytic, "
                   "**green** = benchmark; closer to the diagonal = more accurate.")
    else:
        st.caption(f"Not enough runs with a predicted and real {_unit} for the plot.")

    # ── Formula behind each estimate (the recovered detail table) ───────────────
    if cmp_by_run:
        st.markdown("#### Formula behind each estimate")
        st.caption(f"Per-metric formula + estimated-vs-real, for one **single-GPU** run. "
                   f"It lists the {len(cmp_by_run)} selected single-GPU run(s) whose **batch "
                   f"size was benchmarked** in their model's benchmark report (e.g. the "
                   f"vit_base reports cover batch 48/64/96; a run at a batch the report "
                   f"never measured can't be broken down here). Select more such runs above "
                   f"to see them.")
        pick = st.selectbox("Run (single-GPU)", list(cmp_by_run.keys()),
                            format_func=_short, key="formula_run")
        ftab = cmp_by_run[pick].to_dataframe()
        st.dataframe(ftab, hide_index=True, use_container_width=True)
        _dl_csv(ftab, "benchmark_formulas.csv", "Download formulas")

    # ── Speedup validation when a single + DDP pair of the same model is picked ──
    groups: dict = defaultdict(dict)
    for lbl in sel:
        r = labelled[lbl]
        if r.mode in ("single", "ddp"):
            groups[(r.env, r.model)][r.mode] = r
    sp_lines = []
    for (env, model), d in groups.items():
        if "single" in d and "ddp" in d:
            s, dd = _real_min_per_epoch(d["single"]), _real_min_per_epoch(d["ddp"])
            if s and dd:
                real_sp = s / dd
                m = next((mm for e, mo, mm, _ in parsed if e == env and mo == model),
                         next((mm for e, mo, mm, _ in parsed if mo == model), None))
                pred = None
                if m is not None:
                    scen = parse_ddp_scenarios(m)
                    if not scen.empty and {"n_gpus", "speedup"}.issubset(scen.columns):
                        r2 = scen[scen["n_gpus"] == 2]
                        if not r2.empty:
                            pred = float(r2.iloc[0]["speedup"])
                pre = f"predicted **{pred:.2f}×** · " if pred else ""
                sp_lines.append(f"- **{model.replace('_patch16_224','')}** ({env}): "
                                f"{pre}real **{real_sp:.2f}×** (2 GPUs)")
    if sp_lines:
        st.markdown("**Speedup at 2 GPUs — predicted vs real**")
        st.markdown("\n".join(sp_lines))

    st.divider()
    return st.container()


def render_f1_prediction(meta, selected_run, benchmark_csvs) -> None:
    if not benchmark_csvs:
        st.info("Run the benchmark analysis first.")
    else:
        st.markdown("## Empirical performance prediction")
        pred = meta.get("prediction", {})
        curve_val = meta.get("curve_val_f1", [])
        curve_train = meta.get("curve_train_f1", [])
        curve_epochs = meta.get("curve_epochs", [])

        if not pred:
            st.info(
                "No prediction data in this report. "
                "Regenerate with the current version of benchmark.py."
            )
        else:
            # ── Key prediction metrics ─────────────────────────────────────
            pred_best_f1 = float(pred.get("predicted_best_f1", 0) or 0)
            pred_best_ep = int(float(pred.get("predicted_best_epoch", 0) or 0))
            pred_stop_ep = int(float(pred.get("predicted_early_stop_epoch", 0) or 0))
            confidence = pred.get("confidence", "—")

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Expected Val F1", f"{pred_best_f1:.3f}")
            pc2.metric("Estimated best epoch", pred_best_ep)
            pc3.metric("Estimated early stop", pred_stop_ep)
            pc4.metric("Confidence", confidence)

            # ── Predicted F1 curve ─────────────────────────────────────────
            if curve_val and curve_epochs:
                st.markdown("### Estimated F1 curve")
                _band_by_conf = {"high": 0.020, "medium": 0.035, "low": 0.050}
                uncertainty = _band_by_conf.get(str(confidence).lower(), 0.035)
                st.caption(
                    "**Empirical prior**, not a measurement: the expected Val F1 is anchored "
                    "to documented BigEarthNet-S2 runs of this model family and scaled to the "
                    "dataset size of this report. The band widens as confidence drops "
                    f"(here ±{uncertainty:.3f}, confidence **{confidence}**). For a measured "
                    "estimate use the convergence study below."
                )

                fig_pred = go.Figure()

                fig_pred.add_trace(go.Scatter(
                    x=curve_epochs + curve_epochs[::-1],
                    y=[v + uncertainty for v in curve_val] + [v - uncertainty for v in curve_val[::-1]],
                    fill="toself", fillcolor="rgba(37,99,235,0.1)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Uncertainty (±0.015 F1)",
                    showlegend=True,
                ))

                # Predicted Val F1
                fig_pred.add_trace(go.Scatter(
                    x=curve_epochs, y=curve_val,
                    name="Estimated Val F1",
                    mode="lines", line=dict(color=COLORS[0], width=3),
                ))

                # Predicted Train F1
                if curve_train:
                    fig_pred.add_trace(go.Scatter(
                        x=curve_epochs, y=curve_train,
                        name="Estimated Train F1",
                        mode="lines", line=dict(color=COLORS[0], width=2, dash="dot"),
                        opacity=0.6,
                    ))

                # Mark best epoch
                if pred_best_ep <= max(curve_epochs):
                    best_val = curve_val[pred_best_ep - 1] if pred_best_ep <= len(curve_val) else pred_best_f1
                    fig_pred.add_trace(go.Scatter(
                        x=[pred_best_ep], y=[best_val],
                        name=f"Best epoch ({pred_best_ep})",
                        mode="markers", marker=dict(color="gold", size=14, symbol="star"),
                    ))

                # Mark early stop
                if pred_stop_ep <= max(curve_epochs):
                    fig_pred.add_vline(
                        x=pred_stop_ep, line_dash="dash", line_color=COLORS[3],
                        annotation_text=f"Early stop ~ep{pred_stop_ep}",
                        annotation_position="top right",
                    )

                # Curva real si hay run seleccionado
                if selected_run is not None:
                    try:
                        df_actual_pred = _load_df(
                            str(selected_run.log_path),
                            str(selected_run.epoch_csv_path) if selected_run.epoch_csv_path else None,
                        )
                        if not df_actual_pred.empty and "val_f1" in df_actual_pred.columns:
                            fig_pred.add_trace(go.Scatter(
                                x=df_actual_pred["epoch"].tolist(),
                                y=df_actual_pred["val_f1"].tolist(),
                                name="Real Val F1",
                                mode="lines+markers",
                                line=dict(color=COLORS[1], width=2.5),
                                marker=dict(size=5),
                            ))
                    except Exception:
                        pass

                fig_pred.update_layout(
                    **_base_layout(420, "Validation F1 curve — prediction vs real"),
                    xaxis_title="Epoch",
                    yaxis_title="Val F1 (macro)",
                )
                fig_pred.update_yaxes(range=[0.0, 1.0])
                _show(fig_pred, "f1_prediction")

                if selected_run is not None:
                    st.caption(
                        "Blue line = empirical prior | "
                        "second line = real Val F1 of the selected run | "
                        "star = estimated best epoch"
                    )
                else:
                    st.caption(
                        "Select a run in the sidebar to overlay the real results."
                    )

            # Prediction data as a downloadable table
            if curve_val and curve_epochs:
                pred_curve_df = pd.DataFrame({
                    "epoch": curve_epochs,
                    "val_f1_pred": curve_val,
                    "train_f1_pred": curve_train if curve_train else [None] * len(curve_epochs),
                    "val_f1_upper": [v + uncertainty for v in curve_val],
                    "val_f1_lower": [v - uncertainty for v in curve_val],
                })
                _dl_csv(pred_curve_df, "predicted_f1_curve.csv", "Download predicted curve")
