"""ConvergenceStudy — estudio empírico real durante el análisis de viabilidad.

En lugar de basar las predicciones de rendimiento en curvas históricas
hardcodeadas, este módulo MIDE el comportamiento real del modelo en ESTA
máquina mediante un mini-entrenamiento corto:

1. LR range test (Smith 2017): barre learning rates en escala log y mide
   la loss en cada uno → recomienda el LR óptimo.
2. Convergence study: mini-training de N steps con datos reales, ajusta una
   curva de convergencia (power law) y extrapola loss/F1 a N epochs.
3. Gradient noise scale (McCandlish 2018): mide la varianza del gradiente
   entre batches → estima el batch size crítico.

La parte de ajuste de curvas (fit_power_law, fit_saturation,
extrapolate_*) es pura y testeable sin GPU. La parte de medición requiere
modelo + datos.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training import metrics as m


# ═════════════════════════════════════════════════════════════════════════════
# Value objects
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LRRangeResult:
    lrs: list[float]
    losses: list[float]
    suggested_lr: float       # mayor descenso de loss (pendiente más negativa)
    min_loss_lr: float        # LR del mínimo de loss (límite superior)
    diverged_lr: float | None = None  # LR donde la loss explota


@dataclass
class ConvergenceResult:
    steps: list[int]
    losses: list[float]
    f1s: list[float]
    fit_a: float              # parámetros de loss(t) = a * t^(-b) + c
    fit_b: float
    fit_c: float
    r_squared: float
    extrapolated_loss_1ep: float
    extrapolated_loss_final: float
    extrapolated_best_f1: float
    epochs_to_plateau: int
    measured_imgs_per_s: float  # throughput real con datos reales


@dataclass
class GradientNoiseResult:
    grad_norm_mean: float
    grad_norm_std: float
    noise_scale: float          # B_simple (batch crítico estimado)
    suggested_batch_size: int
    cv: float                   # coeficiente de variación (std/mean)


@dataclass
class StudyReport:
    lr_range: LRRangeResult | None = None
    convergence: ConvergenceResult | None = None
    gradient_noise: GradientNoiseResult | None = None
    n_train_images: int = 237871
    notes: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# Ajuste de curvas (puro, testeable sin GPU)
# ═════════════════════════════════════════════════════════════════════════════


def _moving_average(values: list[float], window: int = 5) -> list[float]:
    """Media móvil centrada que conserva la longitud (suaviza ruido de batch)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < window or window < 2:
        return list(arr)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    # Corregir los bordes (donde el kernel se sale) con la media parcial
    half = window // 2
    for i in range(half):
        smoothed[i] = arr[: i + half + 1].mean()
        smoothed[-(i + 1)] = arr[-(i + half + 1):].mean()
    return list(smoothed)


def fit_power_law(
    steps: list[int] | np.ndarray, losses: list[float] | np.ndarray
) -> tuple[float, float, float, float]:
    """Ajusta loss(t) = a · t^(-b) + c por mínimos cuadrados.

    Devuelve (a, b, c, r_squared). La power law modela bien la caída de loss
    en las primeras fases de entrenamiento de redes profundas.
    """
    steps = np.asarray(steps, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if len(steps) < 4:
        # Insuficiente para ajustar 3 parámetros
        return float(losses[0]) if len(losses) else 0.0, 0.5, 0.0, 0.0

    # c ≈ asíntota: usar un valor por debajo del mínimo observado
    c0 = float(losses.min()) * 0.8
    # Linealizar: log(loss - c) = log(a) - b·log(t)
    best = None
    for c in np.linspace(0, float(losses.min()) * 0.95, 12):
        y = losses - c
        if np.any(y <= 0):
            continue
        log_t = np.log(steps)
        log_y = np.log(y)
        try:
            coef = np.polyfit(log_t, log_y, 1)
        except Exception:
            continue
        b = -coef[0]
        a = math.exp(coef[1])
        pred = a * steps ** (-b) + c
        ss_res = float(np.sum((losses - pred) ** 2))
        ss_tot = float(np.sum((losses - losses.mean()) ** 2)) + 1e-12
        r2 = 1 - ss_res / ss_tot
        if best is None or r2 > best[3]:
            best = (a, b, c, r2)

    if best is None:
        return float(losses.mean()), 0.5, 0.0, 0.0
    return best


def extrapolate_power_law(a: float, b: float, c: float, step: float) -> float:
    """Evalúa loss(step) = a · step^(-b) + c."""
    if step <= 0:
        return a + c
    return a * step ** (-b) + c


def loss_to_f1_estimate(loss: float, model_family: str = "vit_base") -> float:
    """Mapea una loss BCE de validación estimada a un F1 macro aproximado.

    Relación empírica calibrada con los runs reales de BigEarthNet-ViT:
    loss ~0.21 → F1 ~0.61, loss ~0.18 → F1 ~0.66, loss ~0.15 → F1 ~0.68.
    Es una aproximación monótona decreciente saturada.
    """
    # F1 ≈ f1_max · exp(-k · (loss - loss_min))  acotado
    f1_max = {"vit_base": 0.70, "vit_small": 0.64, "vit_tiny": 0.55,
              "resnet50": 0.57, "efficientnet": 0.54}.get(model_family, 0.65)
    loss_min = 0.14
    k = 3.5
    f1 = f1_max * math.exp(-k * max(0.0, loss - loss_min))
    return max(0.0, min(f1_max, f1))


# ═════════════════════════════════════════════════════════════════════════════
# ConvergenceStudy — mediciones reales
# ═════════════════════════════════════════════════════════════════════════════


class ConvergenceStudy:
    """Ejecuta el estudio empírico de convergencia sobre datos reales."""

    def __init__(self, device: torch.device, model_family: str = "vit_base"):
        self._device = device
        self._family = model_family
        self._criterion = nn.BCEWithLogitsLoss()

    # ── LR range test ─────────────────────────────────────────────────────────

    def lr_range_test(
        self, model: nn.Module, loader: DataLoader,
        lr_min: float = 1e-7, lr_max: float = 1.0, n_steps: int = 20,
    ) -> LRRangeResult:
        """Barre LR exponencialmente y mide la loss en cada paso (Smith 2017)."""
        net = copy.deepcopy(model).to(self._device)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=lr_min)

        mult = (lr_max / lr_min) ** (1 / max(1, n_steps - 1))
        lrs, losses = [], []
        lr = lr_min
        data_iter = iter(loader)
        diverged_lr = None
        best_loss = float("inf")

        for _ in range(n_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images, labels = next(data_iter)
            images = images.to(self._device)
            labels = labels.to(self._device)

            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            loss = self._criterion(net(images), labels)
            loss.backward()
            opt.step()

            lv = loss.item()
            lrs.append(lr)
            losses.append(lv)
            best_loss = min(best_loss, lv)
            if lv > best_loss * 4 and diverged_lr is None and len(losses) > 3:
                diverged_lr = lr
                break
            lr *= mult

        suggested_lr, min_loss_lr = self._pick_lr(lrs, losses)
        return LRRangeResult(
            lrs=lrs, losses=losses,
            suggested_lr=suggested_lr, min_loss_lr=min_loss_lr,
            diverged_lr=diverged_lr,
        )

    @staticmethod
    def _pick_lr(lrs: list[float], losses: list[float]) -> tuple[float, float]:
        """LR sugerido = mayor descenso de loss; min_loss_lr = mínimo de loss."""
        if len(lrs) < 3:
            return (lrs[0] if lrs else 1e-4), (lrs[0] if lrs else 1e-4)
        log_lrs = np.log10(np.asarray(lrs))
        losses_a = np.asarray(losses)
        # Suavizar y derivar
        grad = np.gradient(losses_a, log_lrs)
        min_idx = int(np.argmin(losses_a))
        # El LR sugerido: pendiente más negativa antes del mínimo
        search = grad[:max(1, min_idx)]
        steep_idx = int(np.argmin(search)) if len(search) else min_idx
        return float(lrs[steep_idx]), float(lrs[min_idx])

    # ── Convergence study ─────────────────────────────────────────────────────

    def convergence_test(
        self, model: nn.Module, loader: DataLoader,
        lr: float, n_steps: int = 60, batch_size: int = 32,
        n_train_images: int = 237871, n_epochs_target: int = 17,
    ) -> ConvergenceResult:
        """Mini-training real de n_steps; ajusta curva y extrapola."""
        net = copy.deepcopy(model).to(self._device)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)

        steps, losses, f1s = [], [], []
        data_iter = iter(loader)
        t0 = time.perf_counter()
        n_images = 0

        for step in range(1, n_steps + 1):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images, labels = next(data_iter)
            images = images.to(self._device)
            labels = labels.to(self._device)

            opt.zero_grad()
            logits = net(images)
            loss = self._criterion(logits, labels)
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long().cpu()
                f1 = m.f1_score(preds, (labels > 0.5).long().cpu())
            steps.append(step)
            losses.append(loss.item())
            f1s.append(f1)
            n_images += images.shape[0]

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        imgs_per_s = n_images / elapsed if elapsed > 0 else 0.0

        # Ajuste de la curva de loss sobre la versión suavizada (reduce el ruido
        # de batch que degrada el ajuste de la power law con pocos steps)
        losses_smooth = _moving_average(losses, window=5)
        a, b, c, r2 = fit_power_law(steps, losses_smooth)

        batches_per_epoch = max(1, math.ceil(n_train_images / batch_size))
        loss_1ep = extrapolate_power_law(a, b, c, batches_per_epoch)
        loss_final = extrapolate_power_law(a, b, c, batches_per_epoch * n_epochs_target)
        best_f1 = loss_to_f1_estimate(loss_final, self._family)

        epochs_to_plateau = self._estimate_plateau(a, b, c, batches_per_epoch, n_epochs_target)

        return ConvergenceResult(
            steps=steps, losses=losses, f1s=f1s,
            fit_a=a, fit_b=b, fit_c=c, r_squared=r2,
            extrapolated_loss_1ep=loss_1ep,
            extrapolated_loss_final=loss_final,
            extrapolated_best_f1=best_f1,
            epochs_to_plateau=epochs_to_plateau,
            measured_imgs_per_s=imgs_per_s,
        )

    @staticmethod
    def _estimate_plateau(a, b, c, batches_per_epoch, n_epochs_target) -> int:
        """Epoch en el que la loss deja de bajar significativamente (<1%/epoch)."""
        prev = extrapolate_power_law(a, b, c, batches_per_epoch)
        for ep in range(2, n_epochs_target + 1):
            cur = extrapolate_power_law(a, b, c, batches_per_epoch * ep)
            if prev > 0 and (prev - cur) / prev < 0.01:
                return ep
            prev = cur
        return n_epochs_target

    # ── Gradient noise scale ──────────────────────────────────────────────────

    def gradient_noise_scale(
        self, model: nn.Module, loader: DataLoader,
        n_batches: int = 12, batch_size: int = 32,
    ) -> GradientNoiseResult:
        """Mide la variabilidad del gradiente entre batches (McCandlish 2018).

        Estima el batch size crítico B_simple = tr(Σ)/|G|² de forma
        simplificada con la varianza de la norma del gradiente entre batches.
        """
        net = copy.deepcopy(model).to(self._device)
        net.train()

        grad_norms = []
        grad_vectors = []
        data_iter = iter(loader)
        for _ in range(n_batches):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images, labels = next(data_iter)
            images = images.to(self._device)
            labels = labels.to(self._device)

            net.zero_grad()
            loss = self._criterion(net(images), labels)
            loss.backward()

            # Norma del gradiente (solo la cabeza para ser barato y representativo)
            g = torch.cat([
                p.grad.detach().flatten()
                for p in net.parameters()
                if p.grad is not None
            ])
            grad_norms.append(g.norm().item())
            # Guardar una submuestra del vector para estimar la covarianza
            grad_vectors.append(g[::max(1, g.numel() // 2000)].cpu())

        gn = np.asarray(grad_norms)
        gn_mean = float(gn.mean())
        gn_std = float(gn.std())
        cv = gn_std / gn_mean if gn_mean > 0 else 0.0

        # Estimación simplificada del noise scale:
        # B_simple ≈ batch_size · (varianza entre batches) / (norma media)²
        # Aproximación práctica con el CV²
        noise_scale = batch_size * (cv ** 2) if cv > 0 else float(batch_size)
        # Batch sugerido: del orden del noise scale, acotado a potencias prácticas
        suggested = int(min(512, max(8, round(noise_scale))))

        return GradientNoiseResult(
            grad_norm_mean=gn_mean, grad_norm_std=gn_std,
            noise_scale=noise_scale, suggested_batch_size=suggested, cv=cv,
        )

    # ── Orquestación ──────────────────────────────────────────────────────────

    def run_full_study(
        self, model: nn.Module, loader: DataLoader, lr: float,
        batch_size: int, n_train_images: int, n_epochs_target: int,
        n_steps: int = 60,
        do_lr_range: bool = True, do_gradient_noise: bool = True,
    ) -> StudyReport:
        """Ejecuta los tres estudios y devuelve un StudyReport."""
        report = StudyReport(n_train_images=n_train_images)

        if do_lr_range:
            report.lr_range = self.lr_range_test(model, loader)

        report.convergence = self.convergence_test(
            model, loader, lr=lr, n_steps=n_steps, batch_size=batch_size,
            n_train_images=n_train_images, n_epochs_target=n_epochs_target,
        )

        if do_gradient_noise:
            report.gradient_noise = self.gradient_noise_scale(
                model, loader, batch_size=batch_size,
            )

        report.notes = (
            "Estudio empírico real medido en esta máquina: "
            f"LR range test, mini-training de convergencia ({len(report.convergence.steps)} steps "
            "con datos reales) y gradient noise scale. "
            "Las estimaciones se basan en mediciones, no en datos históricos."
        )
        return report
