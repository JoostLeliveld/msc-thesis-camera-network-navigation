#!/usr/bin/env python3
"""What the learned correction removes, on the sealed final audit D_eval (the population
Table I, the fusion results and the correction numbers in the text are quoted on).

(a) Residuals in each camera's ray frame (along the ray, across it), raw projection against
    the correction. The cross marks each cloud's position-balanced mean.
(b) Distribution of position error, raw against corrected, with the position-balanced RMSE
    (RMS over positions of the per-position mean error, as in the audit report) in the legend.

The corrected positions come from pipeline/final_audit_corrected_xy.py, which rebuilds them
from the audit inputs and checks every error against the audit; the RMSEs are checked
against the audit report here. The structured-only ablation exists on D_dev only.

    python3 figures/make_correction.py
"""
from __future__ import annotations

import collections
import json

import numpy as np

import paper as P

VARIANTS = (("raw", "raw projection", "#b3b3b3", "-"),
            ("corrected", "corrected", P.INK, "-"))


def load():
    a = np.load(P.THESIS / "final_audit_corrected_xy.npz", allow_pickle=False)
    truth = a["truth_xy_m"]
    ray = a["raw_xy_m"] - a["camera_xy_m"]
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    to_ray = lambda v: np.c_[(v * ray).sum(1), -ray[:, 1] * v[:, 0] + ray[:, 0] * v[:, 1]]
    return a["position_key"], truth, {"raw": a["raw_xy_m"], "corrected": a["corrected_xy_m"]}, to_ray


def balanced_rmse(keys, err):
    groups = collections.defaultdict(list)
    for k, e in zip(keys, err):
        groups[k].append(e)
    # final_audit.py's definition: RMS over positions of the per-position mean error.
    return float(np.sqrt(np.mean(np.square([np.mean(v) for v in groups.values()]))))


def position_mean(keys, values):
    groups = collections.defaultdict(list)
    for k, v in zip(keys, values):
        groups[k].append(v)
    return np.mean([np.mean(v, axis=0) for v in groups.values()], axis=0)


def main():
    keys, truth, xy, to_ray = load()
    report = json.loads((P.THESIS / "final_audit/report.json").read_text())
    audit = {"raw": report["raw_error"], "corrected": report["corrected_error"]}
    err = {k: np.linalg.norm(v - truth, axis=1) for k, v in xy.items()}
    rmse = {k: balanced_rmse(keys, e) for k, e in err.items()}
    for k, value in rmse.items():   # the instrument must reproduce the manifest exactly
        if abs(value - audit[k]["equal_position_rmse"]) > 1e-9:
            raise RuntimeError(f"{k}: rebuilt RMSE {value} != audit {audit[k]['equal_position_rmse']}")

    fig, (ax, bx) = P.plt.subplots(1, 2, figsize=(P.COLUMN, 1.75), gridspec_kw=dict(width_ratios=(1, 1.25), wspace=0.42))
    for key, label, colour, _ in VARIANTS:
        r = 100.0 * to_ray(xy[key] - truth)
        ax.scatter(r[:, 0], r[:, 1], s=0.6, lw=0, color=colour, alpha=0.5 if key == "raw" else 0.35,
                   rasterized=True, zorder=2 if key == "raw" else 3)
        mean = position_mean(keys, r)
        ax.plot(*mean, "+", color=colour if key != "raw" else "#6f6f6f", ms=7, mew=1.3, zorder=5)
    ax.axhline(0, color=P.MUTED, lw=0.4, zorder=1); ax.axvline(0, color=P.MUTED, lw=0.4, zorder=1)
    ax.set_xlim(-60, 30); ax.set_ylim(-45, 45); ax.set_aspect("equal")
    ax.set_xlabel("along ray (cm)", labelpad=1); ax.set_ylabel("across ray (cm)", labelpad=1)
    bias = position_mean(keys, 100.0 * to_ray(xy["raw"] - truth))
    ax.annotate(f"raw bias\n{bias[0]:+.0f} cm", xy=(bias[0], bias[1]), xytext=(-58, 30), fontsize=6.3,
                color="#5f5f5f", arrowprops=dict(arrowstyle="-", lw=0.5, color="#8f8f8f"))
    fig.text(0.0, 0.97, "(a)", fontweight="bold", va="top")

    grid = np.logspace(np.log10(0.3), np.log10(80), 300)
    for key, label, colour, ls in VARIANTS:
        e = np.sort(100.0 * err[key])
        bx.plot(grid, np.searchsorted(e, grid) / e.size, color=colour, ls=ls, lw=1.3,
                label=f"{label}  {100 * rmse[key]:.1f}")
    bx.set_xscale("log"); bx.set_xlim(0.3, 80); bx.set_ylim(0, 1.0)
    bx.set_xticks([1, 3, 10, 30]); bx.set_xticklabels(["1", "3", "10", "30"])
    bx.set_xlabel("position error (cm)", labelpad=1); bx.set_ylabel("fraction of observations", labelpad=1)
    bx.text(0.36, 0.78, f"corrected\nRMSE {100 * rmse['corrected']:.1f} cm", fontsize=6, ha="left")
    bx.text(23, 0.1, f"raw\nRMSE {100 * rmse['raw']:.1f} cm", fontsize=6, ha="right", color="#6f6f6f")
    fig.text(0.47, 0.97, "(b)", fontweight="bold", va="top")
    P.save(fig, "correction")
    print({k: round(100 * v, 2) for k, v in rmse.items()}, "observations", len(keys), "positions", len(set(keys)))


if __name__ == "__main__":
    main()
