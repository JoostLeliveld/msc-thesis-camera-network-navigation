"""Small deterministic Bernoulli Gaussian-process implementation for availability.

The commissioning data contain many repeated observations along the same route.
They are aggregated into spatial cells as binomial counts, so every opportunity
contributes while the Laplace approximation remains small enough to fit and
serialize without an additional GP runtime dependency.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.linalg import solve_triangular


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def rbf(a: np.ndarray, b: np.ndarray, length_scale_m: float, variance: float) -> np.ndarray:
    """Squared-exponential covariance between planar positions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return float(variance) * np.exp(-0.5 * squared / float(length_scale_m) ** 2)


def aggregate_binomial(
    xy: np.ndarray, labels: np.ndarray, *, cell_size_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate binary opportunities into deterministic planar cells."""
    xy = np.asarray(xy, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(labels):
        raise ValueError("xy must have shape (n, 2) and match labels")
    if not np.isfinite(xy).all() or not np.isfinite(labels).all():
        raise ValueError("GP training data must be finite")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be positive")
    if np.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("GP labels must lie in [0, 1]")

    cells = np.floor(xy / float(cell_size_m) + 0.5).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(float)
    success = np.bincount(inverse, weights=labels).astype(float)
    centres = np.column_stack([
        np.bincount(inverse, weights=xy[:, axis]) / count for axis in range(2)
    ])
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    return centres[order], success[order], count[order]


def fit_laplace_binomial(
    centres_xy_m: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    *,
    length_scale_m: float,
    latent_variance: float = 4.0,
    prior_probability: float | None = None,
    jitter: float = 1e-6,
    maximum_iterations: int = 80,
    tolerance: float = 1e-7,
) -> dict:
    """Fit an RBF GP with a binomial likelihood using a Laplace approximation."""
    centres = np.asarray(centres_xy_m, dtype=float)
    success = np.asarray(successes, dtype=float)
    total = np.asarray(trials, dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("centres_xy_m must have shape (n, 2)")
    if len(centres) == 0 or len(success) != len(centres) or len(total) != len(centres):
        raise ValueError("binomial arrays must be non-empty and have equal length")
    if np.any(total <= 0.0) or np.any(success < 0.0) or np.any(success > total):
        raise ValueError("invalid binomial counts")
    if length_scale_m <= 0.0 or latent_variance <= 0.0 or jitter <= 0.0:
        raise ValueError("GP kernel constants must be positive")

    empirical = float(success.sum() / total.sum())
    base = empirical if prior_probability is None else float(prior_probability)
    base = float(np.clip(base, 1e-5, 1.0 - 1e-5))
    prior_mean = math.log(base / (1.0 - base))
    kernel = rbf(centres, centres, length_scale_m, latent_variance)
    kernel.flat[:: len(centres) + 1] += float(jitter)
    latent = np.full(len(centres), prior_mean, dtype=float)

    converged = False
    for iteration in range(int(maximum_iterations)):
        probability = _sigmoid(latent)
        weight = np.maximum(total * probability * (1.0 - probability), 1e-12)
        root_weight = np.sqrt(weight)
        system = np.eye(len(centres)) + root_weight[:, None] * kernel * root_weight[None, :]
        cholesky = np.linalg.cholesky(system)
        centred = latent - prior_mean
        rhs = weight * centred + success - total * probability
        projected = root_weight * (kernel @ rhs)
        correction = solve_triangular(
            cholesky.T,
            solve_triangular(cholesky, projected, lower=True, check_finite=False),
            lower=False,
            check_finite=False,
        )
        alpha = rhs - root_weight * correction
        updated = prior_mean + kernel @ alpha
        if float(np.max(np.abs(updated - latent))) <= tolerance:
            latent = updated
            converged = True
            break
        latent = updated

    probability = _sigmoid(latent)
    weight = np.maximum(total * probability * (1.0 - probability), 1e-12)
    root_weight = np.sqrt(weight)
    system = np.eye(len(centres)) + root_weight[:, None] * kernel * root_weight[None, :]
    cholesky = np.linalg.cholesky(system)
    alpha = np.linalg.solve(kernel, latent - prior_mean)
    return {
        "centres_xy_m": centres,
        "alpha": alpha,
        "sqrt_weight": root_weight,
        "cholesky": cholesky,
        "prior_mean_logit": prior_mean,
        "length_scale_m": float(length_scale_m),
        "latent_variance": float(latent_variance),
        "jitter": float(jitter),
        "iterations": iteration + 1,
        "converged": converged,
        "training_cells": int(len(centres)),
        "training_opportunities": int(total.sum()),
    }


def predict_laplace(
    model: dict,
    query_xy_m: np.ndarray,
    *,
    uncertainty_penalty: float = 0.0,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return logistic-Gaussian predictive probability and latent standard deviation."""
    query = np.asarray(query_xy_m, dtype=float)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[1] != 2 or not np.isfinite(query).all():
        raise ValueError("query_xy_m must be finite with shape (n, 2)")
    if not math.isfinite(uncertainty_penalty) or uncertainty_penalty < 0.0:
        raise ValueError("uncertainty_penalty must be finite and non-negative")

    centres = np.asarray(model["centres_xy_m"], dtype=float)
    alpha = np.asarray(model["alpha"], dtype=float)
    root_weight = np.asarray(model["sqrt_weight"], dtype=float)
    cholesky = np.asarray(model["cholesky"], dtype=float)
    length_scale = float(model["length_scale_m"])
    variance = float(model["latent_variance"])
    prior_mean = float(model["prior_mean_logit"])
    probabilities = np.empty(len(query), dtype=float)
    standard_deviation = np.empty(len(query), dtype=float)
    for start in range(0, len(query), int(batch_size)):
        stop = min(start + int(batch_size), len(query))
        cross = rbf(centres, query[start:stop], length_scale, variance)
        latent_mean = prior_mean + cross.T @ alpha
        projected = solve_triangular(
            cholesky, root_weight[:, None] * cross, lower=True, check_finite=False
        )
        latent_variance = np.maximum(variance - np.sum(projected * projected, axis=0), 1e-12)
        latent_sd = np.sqrt(latent_variance)
        conservative_mean = latent_mean - float(uncertainty_penalty) * latent_sd
        probabilities[start:stop] = _sigmoid(
            conservative_mean / np.sqrt(1.0 + (math.pi / 8.0) * latent_variance)
        )
        standard_deviation[start:stop] = latent_sd
    return probabilities, standard_deviation


def serializable_model(model: dict) -> dict:
    """Convert a fitted model into deterministic JSON-compatible values."""
    output = {}
    for key, value in model.items():
        output[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return output


def model_from_serialized(payload: dict) -> dict:
    """Validate and materialize a serialized fitted model."""
    output = dict(payload)
    for key in ("centres_xy_m", "alpha", "sqrt_weight", "cholesky"):
        output[key] = np.asarray(payload[key], dtype=float)
    n = len(output["centres_xy_m"])
    if output["centres_xy_m"].shape != (n, 2):
        raise ValueError("invalid GP centre shape")
    if output["alpha"].shape != (n,) or output["sqrt_weight"].shape != (n,):
        raise ValueError("invalid GP vector shape")
    if output["cholesky"].shape != (n, n):
        raise ValueError("invalid GP Cholesky shape")
    finite: Iterable[np.ndarray] = (
        output["centres_xy_m"], output["alpha"], output["sqrt_weight"],
        output["cholesky"],
    )
    if not all(np.isfinite(value).all() for value in finite):
        raise ValueError("non-finite GP parameters")
    if float(output["length_scale_m"]) <= 0.0 or float(output["latent_variance"]) <= 0.0:
        raise ValueError("invalid GP kernel parameters")
    return output
