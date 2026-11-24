"""Analytic estimate of a run's epoch time, fed with what the run recorded (no Streamlit).

The "Estimate vs Benchmark vs Run" tab used to call ``predict()`` with the model name only:
a PANDA run at 512 px was estimated with the FLOPs of 224 px. Here the predictor receives the
run's own setup — resolution, input channels, classes, split sizes, batch, precision, strategy,
GPU count and GPU name — read from its manifest (``run_<ts>.json``) and, for what the manifest
lacks, from the log's ``Configuración:`` line. What is not recorded is not guessed: the estimate
is skipped and says what is missing.

Pure: the predictor and the profile of the model are imported where used (the profile needs
torch, only for models outside the curated table).
"""
from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from src.performance_model import (
    THROUGHPUT_MEASURED, THROUGHPUT_THREE_TERM, ModelSpec, gpu_spec, predict,
)

#: Resolution the predictor's curated table and the default of every dataset assume.
DEFAULT_IMG_SIZE = 224
#: Strategies whose global batch is the per-GPU batch times the GPU count.
_DATA_PARALLEL = ("ddp", "heterogeneous")
_FROM_MANIFEST = "run manifest"

_THROUGHPUT_LABEL = {
    THROUGHPUT_THREE_TERM: "three terms (compute + memory traffic + cost per step)",
    THROUGHPUT_MEASURED: "measured throughput",
}
_FLOPS_ONLY = "FLOPs only (MFU × precision factor)"

DatasetInput = Callable[[str], "tuple[int, int] | None"]
SpecResolver = Callable[[str, int, int, int], "ModelSpec | None"]


def dataset_input(name: str) -> tuple[int, int] | None:
    """``(in_chans, num_classes)`` of a registered dataset, or None if it is unknown.

    Imports the registry lazily (it needs torch); only reads the metadata of the spec."""
    try:
        from src.data.registry import get_dataset_spec
        spec = get_dataset_spec(name)
        return int(spec.in_channels), len(spec.classes)
    except Exception:       # unknown dataset or a spec that cannot be built here
        return None


def resolve_spec(name: str, in_chans: int, num_classes: int, img_size: int) -> ModelSpec | None:
    """Model card for the run's input (curated table at 3×224², otherwise profiled)."""
    from src.model_profile import resolve_model_spec
    return resolve_model_spec(name, in_chans=in_chans, num_classes=num_classes, img_size=img_size)


@dataclass(frozen=True)
class RunEstimateInputs:
    """What the predictor needs about a run; ``None`` fields are what the run did not record."""
    model: str
    dataset: str
    strategy: str
    precision: str
    img_size: int
    in_chans: int | None
    num_classes: int | None
    n_train: int | None
    n_val: int | None
    batch_per_gpu: int | None
    n_gpus: int | None
    gpu: str | None
    gpu_source: str = ""
    notes: tuple[str, ...] = ()
    micro_batches: int | None = None   # GPipe micro-batches (pipeline runs)

    @property
    def missing(self) -> list[str]:
        """Human-readable list of what is needed and not recorded."""
        need = {"GPU name": self.gpu, "dataset size": self.n_train, "batch size": self.batch_per_gpu,
                "number of GPUs": self.n_gpus, "input channels": self.in_chans,
                "number of classes": self.num_classes}
        return [k for k, v in need.items() if v is None]


@dataclass(frozen=True)
class RunEstimate:
    """Predicted against measured epoch time (train + val) of one run."""
    predicted_s: float | None = None
    measured_s: float | None = None
    energy_wh: float | None = None
    throughput_model: str = ""
    gpu: str | None = None
    skipped: str = ""                  # why there is no estimate ("" when there is)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def error_pct(self) -> float | None:
        """Signed (predicted − measured) / measured, in %; None without both."""
        if self.predicted_s is None or not self.measured_s:
            return None
        return (self.predicted_s - self.measured_s) / self.measured_s * 100.0


def measured_epoch_time_s(epoch_times_s: Sequence[float | None]) -> float | None:
    """Median epoch time without the first epoch (it pays the warm-up); with a single epoch,
    that one. None when there is no time."""
    vals = [float(v) for v in epoch_times_s if v is not None and not math.isnan(v) and v > 0]
    if not vals:
        return None
    return statistics.median(vals[1:] if len(vals) > 1 else vals)


def describe_throughput(throughput_model: str) -> str:
    """Where the predicted throughput comes from, in words."""
    return _THROUGHPUT_LABEL.get(throughput_model, _FLOPS_ONLY)


def _int(value) -> int | None:
    m = re.match(r"\d+", str(value)) if value not in (None, "") else None
    return int(m.group()) if m else None


def _ngpus_from_batch(text: str | None) -> int | None:
    """GPU count from ``16/GPU (global=32)`` (global ÷ per-GPU), None if not both."""
    per = _int(text)
    glob = re.search(r"global\s*=?\s*(\d+)", str(text or ""))
    return max(1, round(int(glob.group(1)) / per)) if per and glob else None


def _device_name(manifest: Mapping) -> str | None:
    devices = (manifest.get("env") or {}).get("devices") or []
    return str(devices[0]) if devices else None


def inputs_from_run(*, model: str, dataset: str, mode: str, precision: str | None,
                    manifest: Mapping | None, config_line: Mapping[str, str],
                    fallback_sizes: tuple[int | None, int | None] = (None, None),
                    fallback_gpu: tuple[str, str] | None = None,
                    dataset_info: DatasetInput = dataset_input) -> RunEstimateInputs:
    """Predictor inputs of a run: manifest first, then the ``Configuración:`` line, then the
    caller's fallbacks (``fallback_sizes`` from a benchmark report, ``fallback_gpu`` as
    ``(name, where it comes from)``). The GPU is never guessed from the environment."""
    manifest = manifest or {}
    cfg = manifest.get("config") or {}
    mcfg, dcfg, tcfg = cfg.get("model") or {}, cfg.get("data") or {}, cfg.get("training") or {}
    notes: list[str] = []
    strategy = "heterogeneous" if mode == "ddp_hetero" else mode
    dataset = str(manifest.get("dataset") or config_line.get("dataset") or dataset)

    img_size = _int(mcfg.get("img_size") or dcfg.get("image_size"))
    if img_size is None:
        img_size = DEFAULT_IMG_SIZE
        notes.append(f"resolution not recorded — assumed {DEFAULT_IMG_SIZE} px")

    info = dataset_info(dataset)
    classes = manifest.get("classes") or []
    num_classes = len(classes) if classes else (info[1] if info else None)
    in_chans = info[0] if info else None

    n_train = _int(manifest.get("n_train")) or _int(config_line.get("train")) or fallback_sizes[0]
    n_val = _int(manifest.get("n_val")) or _int(config_line.get("val")) or fallback_sizes[1]
    if n_train and not n_val:
        notes.append("validation size not recorded — about 0.51 × train assumed")

    batch = _int(tcfg.get("batch_size")) or _int(config_line.get("batch"))
    if strategy == "single":
        n_gpus: int | None = 1
    else:
        n_gpus = _int(manifest.get("world_size")) or _ngpus_from_batch(config_line.get("batch"))

    gpu, source = _device_name(manifest), _FROM_MANIFEST
    if gpu is None:
        gpu, source = fallback_gpu if fallback_gpu else (None, "")
    return RunEstimateInputs(
        model=model, dataset=dataset, strategy=strategy, precision=precision or "fp32",
        img_size=img_size, in_chans=in_chans, num_classes=num_classes, n_train=n_train,
        n_val=n_val, batch_per_gpu=batch, n_gpus=n_gpus, gpu=gpu, gpu_source=source,
        notes=tuple(notes), micro_batches=_int(config_line.get("micro_batches")))


def estimate_run(inp: RunEstimateInputs, epoch_times_s: Sequence[float | None],
                 resolve: SpecResolver = resolve_spec) -> RunEstimate:
    """Predicted epoch time of the run against its measured one (median without epoch 1)."""
    measured = measured_epoch_time_s(epoch_times_s)

    def skip(why: str) -> RunEstimate:
        return RunEstimate(measured_s=measured, gpu=inp.gpu, notes=inp.notes, skipped=why)

    if inp.missing:
        return skip("not recorded by the run: " + ", ".join(inp.missing))
    if gpu_spec(inp.gpu) is None:
        return skip(f"GPU '{inp.gpu}' is not in the predictor's table")
    spec = resolve(inp.model, inp.in_chans, inp.num_classes, inp.img_size)
    if spec is None:
        return skip(f"model '{inp.model}' cannot be profiled")
    batch = inp.batch_per_gpu * (inp.n_gpus if inp.strategy in _DATA_PARALLEL else 1)
    extra = {"micro_batches": inp.micro_batches} if inp.micro_batches else {}
    p = predict(inp.strategy, inp.model, inp.gpu, n_gpus=inp.n_gpus, dataset_size=inp.n_train,
                batch=batch, precision=inp.precision, epochs=1, val_size=inp.n_val, model=spec,
                **extra)
    if p is None:
        return skip("the predictor does not model this configuration")
    notes = (*inp.notes, *p.notes)
    if inp.gpu_source != _FROM_MANIFEST:
        notes = (*notes, f"GPU name taken from the {inp.gpu_source}")
    return RunEstimate(predicted_s=p.time_per_epoch_total_s, measured_s=measured,
                       energy_wh=p.energy_per_epoch_wh, throughput_model=p.throughput_model,
                       gpu=inp.gpu, notes=notes)
