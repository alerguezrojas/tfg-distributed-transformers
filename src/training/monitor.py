"""Prediction and layer monitors — what the model decides, epoch by epoch.

Two :class:`~src.training.events.TrainingListener` observers that capture the raw
data the web's Inspector analyses (contract: ``docs/inspeccion.md``):

* :class:`PredictionMonitor` — a fixed, seeded subset of the validation split whose
  probabilities, per-sample loss, penultimate-layer embeddings and Grad-CAM maps are
  saved after every inspected epoch, plus the probabilities of the whole validation
  split (for curves and calibration). Written to ``inspect_<ts>/epoch_<EEE>.npz``.
* :class:`LayerStatsMonitor` — gradient norm, weight norm and update/weight ratio per
  group of layers every ``log_every`` batches, and weight histograms per epoch.

Neither changes the training: no global RNG is consumed (own ``torch.Generator``),
gradients are taken with ``torch.autograd.grad`` w.r.t. activations (parameter
``.grad`` is untouched), the model mode is restored, and the work happens after the
epoch has been timed. Rank 0 only; in DDP the unwrapped module is used so no
collective is ever issued.
"""
from __future__ import annotations

import math
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.training.events import TrainingListener

#: Defaults of the ``monitor:`` config section (docs/inspeccion.md).
MONITOR_DEFAULTS: dict = {
    "enabled": False, "samples": 512, "every": 1, "seed": 0, "thumbnail": 64,
    "embeddings": True, "gradcam": 16, "full_val_max": 4_000_000, "layers": True,
    "system_interval_s": 5,
}


def monitor_config(cfg: dict) -> dict:
    """The ``monitor:`` section of a config with every default filled in."""
    return {**MONITOR_DEFAULTS, **(cfg.get("monitor") or {})}


# ── Model helpers ─────────────────────────────────────────────────────────────

def forward_module(model: nn.Module) -> nn.Module:
    """The callable to run inference with: DDP is unwrapped (a DDP forward on rank 0
    alone could issue collectives); a pipeline wrapper is kept (it places tensors)."""
    from torch.nn.parallel import DistributedDataParallel
    return model.module if isinstance(model, DistributedDataParallel) else model


def core_classifier(model: nn.Module) -> nn.Module:
    """The TimmClassifier inside DDP / ModelParallelViT wrappers (for hooks)."""
    model = forward_module(model)
    return getattr(model, "base", model)


def last_linear(model: nn.Module) -> nn.Linear | None:
    """The last ``nn.Linear`` — its input is the penultimate-layer embedding."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return linears[-1] if linears else None


def cam_target(core: nn.Module):
    """(module, kind, n_prefix) for Grad-CAM, or None.

    ViT/DeiT: input of the last block's attention (``norm1``), tokens (B, N, D)
    without the prefix tokens.
    Other backbones: hooked lazily as the last 4-D activation of the backbone's
    direct children (``kind == "map"``)."""
    backbone = getattr(core, "backbone", None)
    if backbone is None:
        return None
    blocks = getattr(backbone, "blocks", None)
    if isinstance(blocks, nn.Sequential | nn.ModuleList) and len(blocks) and \
            hasattr(backbone, "num_prefix_tokens"):
        # The input of the last block's attention (its norm1), not the block output:
        # with a CLS-token head the last block's patch outputs never reach the logits,
        # so their gradient is zero (same choice as pytorch-grad-cam for ViT).
        last = blocks[-1]
        return getattr(last, "norm1", last), "tokens", int(backbone.num_prefix_tokens)
    children = [m for m in backbone.children()
                if not isinstance(m, nn.Identity | nn.Dropout)]
    return (children, "map", 0) if children else None


def _normalize_maps(cams: torch.Tensor) -> np.ndarray:
    """(P, h, w) float → uint8 0-255, each map scaled to its own maximum."""
    cams = cams.clamp(min=0)
    peak = cams.flatten(1).amax(dim=1).clamp(min=1e-12)
    return (cams / peak[:, None, None] * 255).round().to(torch.uint8).cpu().numpy()


def grad_cam(model: nn.Module, core: nn.Module, images: torch.Tensor,
             classes: torch.Tensor) -> np.ndarray | None:
    """Grad-CAM of ``classes`` for ``images`` (P, h, w) uint8, or None if the
    architecture has no usable layer. Gradients w.r.t. the activation only."""
    target = cam_target(core)
    if target is None:
        return None
    module, kind, n_prefix = target
    captured: list[torch.Tensor] = []

    def keep(_m, _inp, out):
        if isinstance(out, torch.Tensor):
            captured.append(out)

    modules = module if isinstance(module, list) else [module]
    handles = [m.register_forward_hook(keep) for m in modules]
    try:
        with torch.enable_grad():
            logits = model(images)
            if kind == "map":
                captured[:] = [a for a in captured if a.dim() == 4][-1:]
            if not captured:
                return None
            act = captured[-1]
            score = logits.gather(1, classes.to(logits.device)[:, None]).sum()
            (grads,) = torch.autograd.grad(score, act)
    finally:
        for h in handles:
            h.remove()

    act, grads = act.detach().float(), grads.float()
    if kind == "tokens":
        act, grads = act[:, n_prefix:], grads[:, n_prefix:]
        side = math.isqrt(act.shape[1])
        if side * side != act.shape[1]:
            return None
        cams = (act * grads.mean(dim=1, keepdim=True)).sum(-1)
        cams = cams.reshape(-1, side, side)
    else:
        if getattr(core.backbone, "output_fmt", "NCHW") == "NHWC":   # e.g. Swin
            act, grads = act.permute(0, 3, 1, 2), grads.permute(0, 3, 1, 2)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cams = (act * weights).sum(1)
    return _normalize_maps(F.relu(cams))


# ── Subset selection and thumbnails ───────────────────────────────────────────

def select_indices(n: int, m: int, seed: int,
                   labels: torch.Tensor | None = None) -> np.ndarray:
    """``m`` of ``n`` indices with an own generator (the global RNG is untouched).

    With one-hot/multi-hot ``labels`` (n, C), each class first gets up to ``m // C``
    of its samples, the rest is filled at random — rare classes stay represented."""
    g = torch.Generator().manual_seed(seed)
    m = min(m, n)
    perm = torch.randperm(n, generator=g)
    if labels is None or labels.dim() != 2:
        return np.sort(perm[:m].numpy())
    chosen: list[int] = []
    taken = torch.zeros(n, dtype=torch.bool)
    per_class = max(1, m // labels.shape[1])
    for c in range(labels.shape[1]):
        members = perm[labels[perm, c] > 0.5][:per_class]
        members = members[~taken[members]]
        taken[members] = True
        chosen.extend(members.tolist())
    rest = perm[~taken[perm]][: max(0, m - len(chosen))]
    return np.sort(np.array((chosen + rest.tolist())[:m], dtype=np.int64))


def panel_positions(targets: np.ndarray, p: int) -> np.ndarray:
    """``p`` positions of the monitor set, cycling through the classes so every
    class (of the dominant label) appears in the Grad-CAM panel."""
    p = min(p, len(targets))
    if p == 0:
        return np.zeros(0, dtype=np.int64)
    primary = targets.argmax(axis=1)
    queues = [list(np.flatnonzero(primary == c)) for c in range(targets.shape[1])]
    out: list[int] = []
    while len(out) < p and any(queues):
        for q in queues:
            if q and len(out) < p:
                out.append(q.pop(0))
    return np.array(sorted(out), dtype=np.int64)


def thumbnails(images: torch.Tensor, side: int) -> np.ndarray:
    """(B, C, H, W) model inputs → (B, side, side, 3) uint8, each channel scaled to
    its own range (normalization constants are dataset-specific)."""
    x = images[:, :3].float()
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    x = (x - lo) / (hi - lo).clamp(min=1e-6)
    x = F.interpolate(x, size=(side, side), mode="bilinear", antialias=True,
                      align_corners=False)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def per_sample_loss(logits: torch.Tensor, targets: torch.Tensor,
                    sigmoid_outputs: bool) -> torch.Tensor:
    """Unweighted loss of each sample: mean BCE over classes, or cross-entropy."""
    if sigmoid_outputs:
        return F.binary_cross_entropy_with_logits(logits, targets,
                                                  reduction="none").mean(dim=1)
    return F.cross_entropy(logits, targets, reduction="none")


# ── Prediction monitor ────────────────────────────────────────────────────────

class PredictionMonitor(TrainingListener):
    """Saves what the model predicts for a fixed validation subset, every epoch."""

    def __init__(self, trainer, val_dataset, task, classes, sink, log_dir: str | Path,
                 timestamp: str, config: dict, batch_size: int = 64, num_workers: int = 0):
        self._trainer = trainer                  # base trainer: .model, .device
        self._task = task
        self._classes = list(classes)
        self._sink = sink
        self._cfg = {**MONITOR_DEFAULTS, **config}
        self._log_dir = Path(log_dir)
        self.dir = self._log_dir / f"inspect_{timestamp}"
        self._val = val_dataset
        self._batch_size = max(1, min(int(batch_size), 64))
        self._num_workers = num_workers
        self._indices: np.ndarray | None = None
        self._targets: torch.Tensor | None = None
        self._panel = np.zeros(0, dtype=np.int64)
        self._cam_ok = True
        self._warned = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _loader(self) -> DataLoader:
        # Own generator: a DataLoader iterator draws its base seed from the global
        # RNG otherwise, which would shift the training's random stream.
        return DataLoader(Subset(self._val, self._indices.tolist()),
                          batch_size=self._batch_size, shuffle=False,
                          num_workers=self._num_workers,
                          generator=torch.Generator().manual_seed(0))

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self._log_dir))

    def _warn(self, what: str, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            warnings.warn(f"prediction monitor: {what} failed ({exc!r}); the training "
                          f"continues", RuntimeWarning, stacklevel=3)

    # ── TrainingListener ─────────────────────────────────────────────────────

    def on_fit_start(self, epochs):
        try:
            self._prepare()
        except Exception as exc:  # noqa: BLE001 — never stop the training
            self._indices = None
            self._warn("set-up", exc)

    def _prepare(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        labels = getattr(self._val, "targets", None)
        labels = torch.as_tensor(labels) if labels is not None else None
        self._indices = select_indices(len(self._val), int(self._cfg["samples"]),
                                       int(self._cfg["seed"]), labels)
        targets, thumbs = [], []
        side = int(self._cfg["thumbnail"])
        for images, y in self._loader():
            targets.append(self._task.hard_targets(y.float()).to(torch.uint8))
            thumbs.append(thumbnails(images, side))
        self._targets = torch.cat(targets)
        target_np = self._targets.numpy()
        self._panel = panel_positions(target_np, int(self._cfg["gradcam"]))
        model = core_classifier(self._trainer.model)
        head = last_linear(model)
        np.savez_compressed(
            self.dir / "monitor_set.npz", indices=self._indices, targets=target_np,
            thumbs=np.concatenate(thumbs), panel=self._panel,
            classes=np.array(self._classes, dtype=str), task=np.array(self._task.name))
        self._sink.emit("monitor_start", dir=self._rel(self.dir),
                        samples=len(self._indices), panel=len(self._panel),
                        classes=self._classes,
                        full_val=int(self._cfg["full_val_max"]) > 0,
                        embedding_dim=head.in_features if head is not None else None)

    def on_epoch_end(self, epoch, epochs, train_m, val_m, best, is_best, epoch_time_s):
        if self._indices is None:
            return
        every = max(1, int(self._cfg["every"]))
        if epoch % every and epoch != epochs:
            return
        t0 = time.time()
        try:
            path = self._capture(epoch, val_m)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"epoch {epoch} capture", exc)
            return
        self._sink.emit("inspection", epoch=epoch, file=self._rel(path),
                        seconds=round(time.time() - t0, 3))

    def _capture(self, epoch: int, val_m: dict) -> Path:
        model = forward_module(self._trainer.model)
        core = core_classifier(self._trainer.model)
        device = self._trainer.device
        was_training = model.training
        head = last_linear(core) if self._cfg["embeddings"] else None
        feats: list[torch.Tensor] = []
        hook = (head.register_forward_hook(lambda _m, inp, _o: feats.append(inp[0].detach()))
                if head is not None else None)
        probs, losses = [], []
        model.eval()
        try:
            with torch.no_grad():
                for images, y in self._loader():
                    logits = model(images.to(device)).float()
                    y = y.float().to(logits.device)
                    probs.append(self._task.probabilities(logits).cpu())
                    losses.append(per_sample_loss(logits, y, self._task.sigmoid_outputs).cpu())
            arrays = {"probs": torch.cat(probs).numpy().astype(np.float16),
                      "loss": torch.cat(losses).numpy().astype(np.float32)}
            if feats:
                arrays["embeddings"] = torch.cat([f.float().cpu() for f in feats]) \
                    .numpy().astype(np.float16)
            if hook is not None:
                hook.remove()
                hook = None
            if len(self._panel) and self._cam_ok:
                arrays.update(self._cams(model, core, device, arrays["probs"]))
        finally:
            if hook is not None:
                hook.remove()
            model.train(was_training)
        arrays.update(self._full_val(val_m))
        path = self.dir / f"epoch_{epoch:03d}.npz"
        np.savez_compressed(path, **arrays)
        return path

    def _cams(self, model, core, device, probs: np.ndarray) -> dict:
        images = torch.stack([self._val[int(self._indices[p])][0] for p in self._panel])
        classes = torch.as_tensor(probs[self._panel].astype(np.float32).argmax(axis=1))
        maps = grad_cam(model, core, images.to(device), classes)
        if maps is None:
            self._cam_ok = False
            warnings.warn("prediction monitor: Grad-CAM is not available for this "
                          "architecture; maps disabled", RuntimeWarning, stacklevel=2)
            return {}
        return {"cam": maps, "cam_class": classes.numpy().astype(np.int64)}

    def _full_val(self, val_m: dict) -> dict:
        out: dict = {}
        threshold = val_m.get("_optimal_threshold")
        out["threshold"] = np.float32(0.5 if threshold is None else threshold)
        probs, labels = val_m.get("_probs"), val_m.get("_labels")
        if probs is None or labels is None:
            return out
        if probs.numel() > int(self._cfg["full_val_max"]):
            return out
        out["val_probs"] = probs.float().cpu().numpy().astype(np.float16)
        out["val_targets"] = labels.cpu().numpy().astype(np.uint8)
        return out


# ── Layer statistics ──────────────────────────────────────────────────────────

def layer_group(name: str) -> str:
    """Group a parameter name into a readable layer group:
    ``backbone.blocks.3.attn.qkv.weight`` → ``backbone.blocks.3``."""
    parts = name.split(".")
    while parts and parts[0] in ("module", "base"):          # DDP / pipeline wrappers
        parts = parts[1:]
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return ".".join(parts[:2])


class LayerStatsMonitor(TrainingListener):
    """Per-group gradient/weight norms and update ratio (batch hook) and weight
    histograms (epoch end). The update is the step AdamW just applied, rebuilt from
    its state; for other optimizers ``lr × grad`` is used."""

    BINS = 32

    def __init__(self, trainer, sink, log_every: int = 50):
        self._trainer = trainer
        self._sink = sink
        self._log_every = max(1, int(log_every))
        self._epoch = 0
        self._groups: OrderedDict[str, list] | None = None
        self._warned = False

    def _param_groups(self) -> OrderedDict[str, list]:
        if self._groups is None:
            hyper = {}
            for group in self._trainer.optimizer.param_groups:
                for p in group["params"]:
                    hyper[id(p)] = group
            groups: OrderedDict[str, list] = OrderedDict()
            for name, p in self._trainer.model.named_parameters():
                if p.requires_grad:
                    groups.setdefault(layer_group(name), []).append((p, hyper.get(id(p))))
            self._groups = groups
        return self._groups

    def _update(self, p: torch.Tensor, hp: dict | None) -> torch.Tensor | None:
        if hp is None or p.grad is None:
            return None
        state = self._trainer.optimizer.state.get(p, {})
        lr = hp.get("lr", 0.0)
        if "exp_avg" in state and "exp_avg_sq" in state:
            beta1, beta2 = hp.get("betas", (0.9, 0.999))
            step = float(state.get("step", 1))
            m_hat = state["exp_avg"] / (1 - beta1 ** step)
            v_hat = state["exp_avg_sq"] / (1 - beta2 ** step)
            upd = m_hat / (v_hat.sqrt() + hp.get("eps", 1e-8))
            return lr * (upd + hp.get("weight_decay", 0.0) * p.detach())
        return lr * p.grad

    def on_epoch_start(self, epoch, epochs):
        self._epoch = epoch

    def on_batch(self, epoch: int, batch_idx: int, n_batches: int, metrics: dict) -> None:
        if batch_idx % self._log_every and batch_idx != n_batches:
            return
        try:
            self._sink.emit("layers", epoch=self._epoch or epoch, batch=batch_idx,
                            groups=self._stats())
        except Exception as exc:  # noqa: BLE001 — never stop the training
            if not self._warned:
                self._warned = True
                warnings.warn(f"layer monitor failed ({exc!r}); disabled warnings "
                              f"follow silently", RuntimeWarning, stacklevel=2)

    @torch.no_grad()
    def _stats(self) -> list[dict]:
        # Squared norms are accumulated on each parameter's device and copied to the
        # CPU once per (group, device): a handful of syncs instead of one per tensor.
        per_group = []
        for name, params in self._param_groups().items():
            acc: dict[torch.device, torch.Tensor] = {}
            for p, hp in params:
                w = p.detach().float()
                zero = w.new_zeros(())
                grad = p.grad.detach().float().pow(2).sum() if p.grad is not None else zero
                upd = self._update(p, hp)
                row = torch.stack([w.pow(2).sum(), grad,
                                   upd.float().pow(2).sum() if upd is not None else zero])
                acc[w.device] = acc[w.device] + row if w.device in acc else row
            per_group.append((name, acc))
        rows = []
        for name, acc in per_group:
            w2, g2, u2 = (sum(v.cpu() for v in acc.values()).tolist() if acc
                          else (0.0, 0.0, 0.0))
            wn = math.sqrt(w2)
            rows.append({"name": name, "grad_norm": math.sqrt(g2), "weight_norm": wn,
                         "update_ratio": math.sqrt(u2) / wn if wn > 0 else None})
        return rows

    def on_epoch_end(self, epoch, epochs, train_m, val_m, best, is_best, epoch_time_s):
        try:
            self._sink.emit("histograms", epoch=epoch, groups=self._histograms())
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                self._warned = True
                warnings.warn(f"layer histograms failed ({exc!r})", RuntimeWarning,
                              stacklevel=2)

    @torch.no_grad()
    def _histograms(self) -> list[dict]:
        rows = []
        for name, params in self._param_groups().items():
            weights = [p.detach().float() for p, _ in params]
            lo = min(w.min().item() for w in weights)
            hi = max(w.max().item() for w in weights)
            if hi <= lo:
                hi = lo + 1e-6
            counts = sum(torch.histc(w, bins=self.BINS, min=lo, max=hi).cpu()
                         for w in weights)
            edges = torch.linspace(lo, hi, self.BINS + 1)
            rows.append({"name": name, "edges": edges.tolist(),
                         "counts": counts.long().tolist()})
        return rows
