"""What a run IS, for the web: dataset, task, main metric, loss, seed, hospital, ….

One reader for every page. The metadata comes from, in this order of preference:

1. the run manifest ``run_<ts>.json`` (small; committed with the logs),
2. the ``run_start`` event of ``events_<ts>.jsonl`` (same fields, but usually not
   committed),
3. the ``Configuración:`` line of the training log (what older runs recorded: seed,
   loss, precision, batch, lr, epochs and the train/val sizes).

Each later source only fills what the earlier ones left out, and the dataset and
strategy fall back to the folder layout. Nothing is invented: an unknown field stays
``None``. Pure (no Streamlit, no torch).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.run_manifest import loss_name, read_config_line, read_manifest, read_run_start

#: Seed of the configs: runs with it do not mention it in their label.
DEFAULT_SEED = 42
#: timm's default input size: runs at another resolution mention it.
DEFAULT_IMG_SIZE = 224

# Tasks that the folder name alone identifies (the web never imports the dataset registry,
# which needs torch). Anything else (the MedMNIST subsets) keeps ``task=None`` unless the
# manifest or the events say it.
_KNOWN_TASK = {"bigearthnet": ("multilabel", False), "panda": ("multiclass", True)}
_DEFAULT_LOSS = {"multilabel": "bce", "binary": "bce", "multiclass": "ce"}
_LOSS_LABEL = {"bce": "BCE", "ce": "CE", "ce_emd": "CE + EMD", "focal": "Focal"}


# ── Main metric ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricInfo:
    """A validation metric of a run: its selection key, epoch-CSV column and names."""

    key: str          # value of training.select_by
    column: str       # column of the epoch table
    name: str         # short name (KPI titles, column headers)
    long_name: str    # axis / chart title


F1 = MetricInfo("f1", "val_f1", "F1", "F1 (macro)")
AUC = MetricInfo("auc", "val_auc", "AUC", "ROC AUC (macro)")
QWK = MetricInfo("qwk", "val_qwk", "QWK", "Quadratic weighted kappa")
F1_OPT = MetricInfo("f1_optimal", "f1_at_threshold", "F1 @ opt. threshold",
                    "F1 (macro) at the optimal threshold")
_METRICS = {m.key: m for m in (F1, AUC, QWK, F1_OPT)}


def _default_metric_key(dataset: str, ordinal: bool | None) -> str:
    """The metric a dataset is judged by when the run does not say (PANDA: QWK, the
    official one; MedMNIST: AUC, its reference; otherwise macro F1)."""
    if ordinal or dataset == "panda":
        return "qwk"
    if dataset.startswith("medmnist"):
        return "auc"
    return "f1"


def metric_info(key: str | None, dataset: str = "", ordinal: bool | None = None) -> MetricInfo:
    """MetricInfo of a ``select_by`` key; unknown/None → the dataset's default."""
    return _METRICS.get(str(key or "").lower()) or _METRICS[_default_metric_key(dataset, ordinal)]


def has_values(df, column: str) -> bool:
    return df is not None and column in df.columns and bool(df[column].notna().any())


def resolve_metric(metric: MetricInfo, df) -> MetricInfo:
    """``metric`` if the run's epoch table has it, else macro F1 (older runs did not log
    the AUC or the QWK, so the main metric of their dataset can be missing)."""
    return metric if has_values(df, metric.column) else F1


# ── RunMeta ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunMeta:
    dataset: str
    task: str | None = None
    ordinal: bool | None = None
    classes: tuple[str, ...] = ()
    select: str | None = None          # training.select_by as run (None = unknown)
    loss: str | None = None            # 'bce' | 'ce' | 'ce_emd' | 'focal' …
    emd_weight: float | None = None
    ousm_k: int | None = None          # OUSM: highest-loss samples dropped per batch
    seed: int | None = None
    providers_train: tuple[str, ...] = ()
    providers_val: tuple[str, ...] = ()
    providers_test: tuple[str, ...] = ()
    color_jitter: float | None = None
    img_size: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    epochs: int | None = None
    precision: str | None = None
    n_train: int | None = None
    n_val: int | None = None
    command: str | None = None
    source: str = "path"               # manifest | events | log | path

    @property
    def metric(self) -> MetricInfo:
        """The run's main (selection) metric."""
        return metric_info(self.select, self.dataset, self.ordinal)

    @property
    def default_loss(self) -> str | None:
        return _DEFAULT_LOSS.get(self.task or "")

    @property
    def effective_loss(self) -> str | None:
        """Loss in effect: the configured one, else the task's default."""
        return self.loss or self.default_loss

    @property
    def loss_label(self) -> str:
        """Readable loss name ('BCE', 'CE + EMD'); 'Loss' when it is not known."""
        eff = self.effective_loss
        return _LOSS_LABEL.get(eff, eff.upper()) if eff else "Loss"

    @property
    def providers_label(self) -> str | None:
        """'train: karolinska' (+ ' · val: …' when it differs) if the run restricts providers."""
        if not (self.providers_train or self.providers_val):
            return None
        tr = "+".join(self.providers_train)
        out = f"train: {tr}" if tr else ""
        if self.providers_val and self.providers_val != self.providers_train:
            out += f"{' · ' if out else ''}val: {'+'.join(self.providers_val)}"
        return out

    def tags(self, *, with_seed: bool = True) -> list[str]:
        """What tells the run apart from its neighbours; default values are left out."""
        out: list[str] = []
        if self.loss and self.loss != self.default_loss and self.loss not in ("bce", "ce"):
            out.append(self.loss)
        if self.ousm_k:
            out.append(f"OUSM {self.ousm_k}")
        if with_seed and self.seed not in (None, DEFAULT_SEED):
            out.append(f"seed {self.seed}")
        if (p := self.providers_label):
            out.append(p)
        if self.color_jitter:
            out.append(f"color {self.color_jitter:g}")
        if self.img_size and self.img_size != DEFAULT_IMG_SIZE:
            out.append(f"{self.img_size}px")
        return out

    def group_key(self, run_id: str) -> tuple:
        """Runs with equal keys differ at most in the seed. A run that does not record its
        seed never joins a group (its key is its own id)."""
        if self.seed is None:
            return ("single", run_id)
        return (self.dataset, self.loss, self.emd_weight, self.ousm_k, self.providers_train,
                self.providers_val, self.color_jitter, self.img_size, self.batch_size,
                self.lr, self.epochs, self.precision, self.select)


def _num(value: Any, cast=float) -> Any:
    try:
        return cast(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _providers(cfg: Mapping, split: str) -> tuple[str, ...]:
    prov = (cfg.get("data") or {}).get("providers") or {}
    value = prov.get(split) if isinstance(prov, Mapping) else None
    return tuple(str(v) for v in value) if value else ()


def _from_run_start(d: Mapping, source: str, fallback_dataset: str) -> RunMeta:
    """RunMeta from a manifest / ``run_start`` dict (same field names)."""
    cfg = d.get("config") or {}
    data, model, train = cfg.get("data") or {}, cfg.get("model") or {}, cfg.get("training") or {}
    dataset = str(d.get("dataset") or data.get("name") or fallback_dataset)
    known_task, known_ordinal = _KNOWN_TASK.get(dataset, (None, None))
    ordinal = d.get("ordinal")
    return RunMeta(
        dataset=dataset,
        task=d.get("task") or known_task,
        ordinal=known_ordinal if ordinal is None else bool(ordinal),
        classes=tuple(str(c) for c in d.get("classes") or ()),
        select=str(d.get("select_metric") or train.get("select_by") or "").lower() or None,
        loss=loss_name(train.get("loss")),
        emd_weight=_num(train.get("emd_weight")),
        ousm_k=_num(train.get("ousm_k"), int) or None,
        seed=_num(train.get("seed"), int),
        providers_train=_providers(cfg, "train"), providers_val=_providers(cfg, "val"),
        providers_test=_providers(cfg, "test"),
        color_jitter=_num(data.get("color_jitter")),
        img_size=_num(model.get("img_size") or data.get("image_size"), int),
        batch_size=_num(train.get("batch_size"), int),
        lr=_num(train.get("lr")),
        epochs=_num(d.get("epochs") or train.get("epochs"), int),
        precision=train.get("precision"),
        n_train=_num(d.get("n_train"), int), n_val=_num(d.get("n_val"), int),
        command=d.get("command"), source=source)


def _merge_log(meta: RunMeta, log_path: Path | None) -> RunMeta:
    """Fill the gaps of ``meta`` from the log's ``Configuración:`` line."""
    cfg = read_config_line(log_path) if log_path else {}
    if not cfg:
        return meta
    updates: dict[str, Any] = {}

    def fill(field: str, value: Any) -> None:
        if value is not None and getattr(meta, field) in (None, ()):
            updates[field] = value

    fill("seed", _num(cfg.get("seed") if cfg.get("seed") != "None" else None, int))
    fill("loss", loss_name(cfg.get("loss")))
    fill("precision", cfg.get("precision"))
    fill("lr", _num(cfg.get("lr")))
    fill("epochs", _num(cfg.get("epochs"), int))
    fill("n_train", _num(cfg.get("train"), int))
    fill("n_val", _num(cfg.get("val"), int))
    batch = (cfg.get("batch") or "").split("/")[0]
    fill("batch_size", _num(batch, int))
    if not updates:
        return meta
    from dataclasses import replace
    return replace(meta, **updates)


def load_run_meta(run) -> RunMeta:
    """Metadata of a :class:`~src.web.run_registry.RunInfo` (see the module docstring)."""
    return resolve_meta(
        manifest=read_manifest(getattr(run, "manifest_path", None)),
        run_start=read_run_start(getattr(run, "events_path", None)),
        log_path=run.log_path, dataset=run.dataset)


def resolve_meta(*, manifest: Mapping | None, run_start: Mapping | None,
                 log_path: Path | None, dataset: str) -> RunMeta:
    """The three-source merge of :func:`load_run_meta`, on already-read inputs."""
    if manifest:
        meta = _from_run_start(manifest, "manifest", dataset)
    elif run_start:
        meta = _from_run_start(run_start, "events", dataset)
    else:
        known_task, known_ordinal = _KNOWN_TASK.get(dataset, (None, None))
        meta = RunMeta(dataset=dataset, task=known_task, ordinal=known_ordinal,
                       source="log" if log_path and read_config_line(log_path) else "path")
    return _merge_log(meta, log_path)
