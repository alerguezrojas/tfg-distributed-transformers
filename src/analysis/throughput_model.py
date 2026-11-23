"""Two-term training-throughput model: compute (FLOPs) + memory traffic (saved activations).

Seconds per training image on one GPU::

    t = a · F + b · A          (+ d / B, a fixed cost per step of B images)

with ``F`` the forward GFLOPs per image (multiply-add = 1, the unit of the predictor's
MODEL_TABLE) and ``A`` the GB that autograd saves for the backward pass of one image
(``src.model_profile.saved_activation_gb``, fp32). ``a`` (s per GFLOP) is the inverse of an
effective compute rate and ``b`` (s per GB) the inverse of an effective memory rate; both
are fitted per GPU and precision from measured throughput (``paravit benchmark``).

The FLOPs-only model (``t = a · F``) is what the analytic predictor uses today; on ViTs both
agree, but hierarchical transformers and (above all) depthwise CNNs move much more memory
per FLOP. Study and numbers in docs/performance_model.md ("Arquitecturas distintas de ViT").

Pure NumPy (no torch): unit-testable without hardware.
"""
from __future__ import annotations

import numpy as np


def fit_two_term(flops, saved_gb, ips) -> tuple[float, float]:
    """Least-squares ``(a, b)`` of ``1/ips = a·F + b·A`` (no intercept)."""
    X = np.c_[np.asarray(flops, float), np.asarray(saved_gb, float)]
    y = 1.0 / np.asarray(ips, float)
    if len(y) < 2:
        raise ValueError("need at least two measurements to fit two terms")
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(a), float(b)


def fit_flops_only(flops, ips) -> float:
    """Least-squares ``a`` of ``1/ips = a·F`` (the predictor's current form)."""
    f = np.asarray(flops, float)
    y = 1.0 / np.asarray(ips, float)
    return float(f @ y / (f @ f))


def fit_with_overhead(flops, saved_gb, ips, batch) -> tuple[float, float, float]:
    """Least-squares ``(a, b, d)`` of ``1/ips = a·F + b·A + d/B``: the two terms plus a fixed
    cost ``d`` (seconds) per optimisation step of ``B`` images (kernel launches, optimizer,
    Python), which only weighs when the images are small."""
    f, s, b = (np.asarray(v, float) for v in (flops, saved_gb, batch))
    y = 1.0 / np.asarray(ips, float)
    if len(y) < 3:
        raise ValueError("need at least three measurements to fit three terms")
    (a, bb, d), *_ = np.linalg.lstsq(np.c_[f, s, 1.0 / b], y, rcond=None)
    return float(a), float(bb), float(d)


def predict_ips(a: float, b: float, flops, saved_gb, d: float = 0.0, batch=1) -> np.ndarray:
    """Images per second predicted by ``t = a·F + b·A + d/B`` (``d = 0``: the two terms)."""
    t = a * np.asarray(flops, float) + b * np.asarray(saved_gb, float)
    return 1.0 / (t + d / np.asarray(batch, float))


def relative_errors(pred, meas) -> np.ndarray:
    """``pred/meas − 1`` (positive = predicted faster than measured)."""
    return np.asarray(pred, float) / np.asarray(meas, float) - 1.0


def leave_one_out(flops, saved_gb, ips, two_term: bool = True) -> np.ndarray:
    """Relative error of each measurement predicted by a fit WITHOUT it."""
    f, s, r = (np.asarray(v, float) for v in (flops, saved_gb, ips))
    out = np.empty(len(r))
    for i in range(len(r)):
        keep = np.arange(len(r)) != i
        if two_term:
            a, b = fit_two_term(f[keep], s[keep], r[keep])
        else:
            a, b = fit_flops_only(f[keep], r[keep]), 0.0
        out[i] = relative_errors(predict_ips(a, b, f[i], s[i]), r[i])
    return out


def transfer(a: float, b: float, compute_ratio: float, bandwidth_ratio: float) -> tuple[float, float]:
    """``(a, b)`` on another GPU from the spec sheet alone.

    ``compute_ratio`` = (TFLOPS·π) of the calibration GPU / that of the target GPU, and
    ``bandwidth_ratio`` = memory bandwidth of the calibration GPU / that of the target: a
    slower target has a larger ``a`` (compute) and ``b`` (memory)."""
    if compute_ratio <= 0 or bandwidth_ratio <= 0:
        raise ValueError("ratios must be positive")
    return a * compute_ratio, b * bandwidth_ratio


def common_factor(rel_err) -> float:
    """Geometric-mean factor shared by all errors, ``exp(mean(log(1 + e)))``: what the
    spec-sheet transfer misses for every model alike on that GPU. It is NOT necessarily a
    clock: the SM clock only scales the compute term, and the residual can sit in the
    memory term too (effective bandwidth under a power cap…)."""
    return float(np.exp(np.mean(np.log1p(np.asarray(rel_err, float)))))


def shape_errors(rel_err) -> np.ndarray:
    """Errors once the common factor is removed: how well the RELATIVE cost of the
    architectures is predicted, independently of the GPU's absolute speed."""
    e = np.asarray(rel_err, float)
    return (1.0 + e) / common_factor(e) - 1.0
