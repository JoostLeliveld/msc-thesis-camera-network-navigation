#!/usr/bin/env python3
"""Diagnostics of the corrected measurement error.

(a) Quantiles of the squared Mahalanobis distance (NIS) of every admitted test-set observation
    against the chi-square(2) quantiles a Gaussian error with the fitted covariance would give,
    per covariance model (logs/thesis/final_audit/evaluated.jsonl). On the diagonal the model is
    calibrated at every level; above it the model is overconfident there.
(b) Correlation of one camera's whitened error between two of its frames against the time
    between them, along the driven campaign routes, while moving and with the robot stationary
    (logs/thesis/analysis/temporal_correlation.json, pipeline/analyze_temporal_correlation.py).

Writes separate main-text and appendix figures.

    python3 figures/make_error_structure.py
"""
from __future__ import annotations

import json
import math

import numpy as np

import paper as P

KEYS = (("global", "R0_global_full"), ("per_camera", "R1_per_camera_full"),
        ("spatial", "R2_spatial_full"), ("rproj", "Rproj"))


def main():
    audit = [json.loads(line) for line in (P.THESIS / "final_audit/evaluated.jsonl").open()]
    audit = [r for r in audit if r["outcome"] == "admitted"]
    temporal = json.loads((P.THESIS / "analysis/temporal_correlation.json").read_text())

    fig, ax = P.plt.subplots(1, 1, figsize=(P.COLUMN, 2.35))

    probs = np.linspace(0.01, 0.999, 400)
    theory = -2.0 * np.log(1.0 - probs)
    for name, key in KEYS:
        d2 = np.array([r["mahalanobis_d2"][key] for r in audit])
        ax.plot(theory, np.quantile(d2, probs), color=P.MODEL_COLOUR[name], lw=1.4,
                ls="--" if name == "rproj" else "-", label=P.MODEL_LABEL[name])
    ax.plot([0.02, 60], [0.02, 60], color=P.MUTED, lw=0.8, zorder=0)
    for p in (0.5, 0.95, 0.99, 0.999):
        q = -2.0 * math.log(1.0 - p)
        ax.axvline(q, color=P.MUTED, lw=0.4, ls=":", zorder=0)
        ax.text(q, 0.025, f"{100 * p:g}%", fontsize=6, color=P.MUTED, rotation=90, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.02, 20); ax.set_ylim(0.02, 120)
    ax.set_xlabel(r"$\chi^2_2$ quantile (Gaussian error)")
    ax.set_ylabel("NIS quantile, test set")
    ax.set_title("Error distribution", fontsize=8, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    P.save(fig, "error_distribution")

    fig, bx = P.plt.subplots(1, 1, figsize=(P.COLUMN, 2.35))
    bins = temporal["bins"]
    mid = [0.5 * sum(b["gap_s"]) for b in bins]
    moving = [np.mean(b["corr_whitened_xy"]) for b in bins]
    bx.plot(mid, moving, "o-", color="#56B4E9", ms=3, lw=1.3, label="driving (all models)")
    still = [(m, np.mean(b["stationary_corr_whitened_xy"])) for m, b in zip(mid, bins)
             if b["stationary_corr_whitened_xy"] and b["stationary_pairs"] >= 50]
    bx.plot(*zip(*still), "s-", color=P.INK, ms=3, lw=1.1, label="robot stationary")
    for m, c, bn in list(zip(mid, moving, bins))[::2]:
        bx.annotate(f"{bn['median_displacement_m']:.1f} m", (m, c), xytext=(2, -9), textcoords="offset points",
                    fontsize=6, color="#2b7fb0")
    bx.axhline(0, color=P.MUTED, lw=0.6)
    bx.set_xscale("log"); bx.set_ylim(-0.05, 1.0)
    bx.set_xlabel("time between frames (s)")
    bx.set_ylabel("correlation of whitened error")
    bx.set_title("Between-frame correlation (labels: distance driven)", fontsize=8, loc="left")
    bx.legend(fontsize=6.5, frameon=False, loc="upper right")
    P.save(fig, "temporal_correlation")


if __name__ == "__main__":
    main()
