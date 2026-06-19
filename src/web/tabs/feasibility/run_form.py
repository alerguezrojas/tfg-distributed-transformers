"""Feasibility — run_form."""
from __future__ import annotations

import shlex
import subprocess

import streamlit as st

from src.web.ui.helpers import (ROOT, _get_configs, _get_feasibility_csvs)


def render_run_form() -> None:
    st.subheader("Run feasibility analysis")
    configs_available = _get_configs()
    model_options_f = [
        "vit_base_patch16_224", "vit_tiny_patch16_224",
        "vit_small_patch16_224", "resnet50", "efficientnet_b0",
    ]
    from src.gpu_specs import detect_all
    _gpus_avail = detect_all()
    with st.form("feasibility_form"):
        fa1, fa2 = st.columns(2)
        with fa1:
            feas_model = st.selectbox("Model", model_options_f)
            feas_batches = st.multiselect("Batch sizes", [16, 32, 64, 128], default=[32, 64])
            feas_epochs = st.number_input("Epochs for estimate", min_value=1, value=30)
            feas_dataset_path = st.text_input(
                "Dataset path (optional — to measure real I/O)",
                placeholder="/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2",
            )
        with fa2:
            feas_traces = st.multiselect("Trace modes", ["off", "simple", "deep"],
                                          default=["off", "simple"])
            feas_nfs = st.slider("NFS factor", 1.0, 2.0, 1.0, 0.05,
                                 help="Correction for NFS latency (Verode: ~1.3)")
            feas_config = st.selectbox(
                "YAML config (optional)",
                ["(none)"] + (configs_available if configs_available else []),
            )
            feas_no_disk = st.checkbox("Skip I/O measurement (faster)", value=False)
            feas_study = st.checkbox(
                "Real empirical study (mini-training + LR range + gradient noise)",
                value=False,
                help="Measures real convergence on this machine. Slower (~3-8 min).",
            )
            feas_study_steps = st.number_input(
                "Mini-training steps", min_value=20, max_value=200, value=60,
                help="Only if the empirical study is enabled",
            )
            feas_device = 0
            if len(_gpus_avail) > 1:
                _dev_labels = {
                    f"cuda:{g.index} — {g.name} ({g.cuda_cores:,} CUDA cores)": g.index
                    for g in _gpus_avail
                }
                _sel = st.selectbox("GPU device", list(_dev_labels.keys()),
                                    help="Which GPU to run the benchmark on (multi-GPU host).")
                feas_device = _dev_labels[_sel]
            elif len(_gpus_avail) == 1:
                st.caption(f"GPU: cuda:0 — {_gpus_avail[0].name}")

            # Precision = Tensor-core switch (options gated by the GPU)
            from src.precision import available_precisions, label as _plabel
            _cc = _gpus_avail[0].compute_capability if _gpus_avail else None
            _precs = available_precisions(_cc, is_cuda=bool(_gpus_avail))
            feas_precision = st.selectbox(
                "Precision (Tensor-core switch)", _precs,
                format_func=_plabel,
                help="fp32 = CUDA cores; tf32/amp/bf16 = Tensor cores (faster, less VRAM).",
            )
            feas_compare_prec = st.checkbox(
                "Compare FP32 vs Tensor cores", value=False,
                help="Run an extra FP32-vs-Tensor pass and report the speedup.",
                disabled=len(_precs) <= 1,
            )
        submitted_feas = st.form_submit_button("Run")

    if submitted_feas:
        if not feas_batches:
            st.error("Select at least one batch size.")
        else:
            # Build an argv LIST and run WITHOUT shell=True so free-text fields
            # (e.g. the dataset path) can never be interpreted as shell syntax.
            argv = [
                "uv", "run", "python", "scripts/check_feasibility.py",
                "--model", feas_model,
                "--batch-sizes", *[str(b) for b in feas_batches],
                "--epochs", str(feas_epochs),
                "--trace-modes", *(feas_traces if feas_traces else ["off"]),
            ]
            if feas_nfs != 1.0:
                argv += ["--nfs-factor", str(feas_nfs)]
            if feas_config != "(none)":
                argv += ["--config", f"configs/{feas_config}"]
            if feas_dataset_path.strip():
                argv += ["--dataset-path", feas_dataset_path.strip()]
            if feas_no_disk:
                argv.append("--no-disk-profile")
            if feas_study:
                argv += ["--convergence-study", "--study-steps", str(feas_study_steps)]
            if feas_device:
                argv += ["--device", str(feas_device)]
            if feas_precision and feas_precision != "fp32":
                argv += ["--precision", feas_precision]
            if feas_compare_prec:
                argv.append("--compare-precision")
            st.code(" ".join(shlex.quote(a) for a in argv), language="bash")
            out_ph = st.empty()
            with st.spinner("Running full analysis…"):
                result = subprocess.run(argv, capture_output=True, text=True, cwd=str(ROOT))
            if result.returncode == 0:
                st.success("Analysis complete.")
                out_ph.code(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
                _get_feasibility_csvs.clear()
            else:
                st.error("Error during the analysis:")
                out_ph.code(result.stderr[-2000:])

