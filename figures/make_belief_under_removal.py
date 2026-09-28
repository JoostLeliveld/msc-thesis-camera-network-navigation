#!/usr/bin/env python3
"""Why the removal runs fail: the belief loses its cameras on the routes the global and
per-camera models keep; the spatial model's route keeps them.

One point per run (5 tasks x 3 seeds per arm), after the first command: (a) the peak of the
belief's own one-sigma (major axis), (b) the peak of its true error against ground truth.
Filled: the run succeeded; x: it failed. The horizontal bar is
the arm median.

    python3 figures/make_belief_under_removal.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

import paper as P


def peak(run_dir, column, first, stop):
    best = 0.0
    with (P.REPO / run_dir / "experiment.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            try:
                stamp, value = float(r["stamp"]), float(r[column])
            except (TypeError, ValueError):
                continue
            if first <= stamp <= stop and np.isfinite(value):
                best = max(best, value)
    return best


def main():
    with (P.ANALYSIS / "runs.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    for r in runs:
        s = json.loads((P.REPO / r["run_dir"] / "run_summary.json").read_text())
        first, stop = float(s["first_cmd_stamp"]), float(s["stop_stamp"])
        r["peak_sigma"] = peak(r["run_dir"], "state_sigma_major_m", first, stop)
        r["peak_error"] = peak(r["run_dir"], "belief_error_gt_m", first, stop)
    fig, axes = P.plt.subplots(1, 2, figsize=(P.COLUMN, 1.95), gridspec_kw=dict(wspace=0.42))
    rng = np.random.default_rng(3)
    for ax, key, label in ((axes[0], "peak_sigma", "peak belief one-sigma (m)"),
                           (axes[1], "peak_error", "peak belief error (m)")):
        for k, model in enumerate(P.MODELS):
            for s_i, state in enumerate(("intact", "removal")):
                x = k * 2.6 + s_i
                arm = [r for r in runs if r["model"] == model and r["state"] == state]
                vals = np.array([r[key] for r in arm])
                jitter = rng.uniform(-0.22, 0.22, len(arm))
                for r, v, j in zip(arm, vals, jitter):
                    ok = r["success"] == "1"
                    ax.plot(x + j, v, "o" if ok else "x", ms=3.2 if ok else 3.6,
                            color=P.MODEL_COLOUR[model], alpha=0.9 if state == "removal" else 0.55,
                            mfc=P.MODEL_COLOUR[model] if (ok and state == "removal") else ("white" if ok else None),
                            mew=0.8 if ok else 1.0, zorder=3)
                ax.plot([x - 0.32, x + 0.32], [np.median(vals)] * 2, color=P.INK, lw=1.2, zorder=4)
        ax.set_yscale("log"); ax.set_ylim(0.02, 8.0)
        ax.set_yticks([0.03, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
        ax.minorticks_off()
        ax.set_xticks([k * 2.6 + s for k in range(3) for s in (0, 1)])
        ax.set_xticklabels(["all", "drop."] * 3, fontsize=6.3)
        for k, model in enumerate(P.MODELS):
            ax.text(k * 2.6 + 0.5, -0.2, P.MODEL_LABEL[model], transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=6.5, color=P.MODEL_COLOUR[model], fontweight="bold")
        ax.set_xlim(-0.6, 2 * 2.6 + 1.6)
        ax.set_ylabel(label, labelpad=1)
    fig.text(0.0, 0.98, "(a)", fontweight="bold", va="top")
    fig.text(0.51, 0.98, "(b)", fontweight="bold", va="top")
    P.save(fig, "belief_under_removal")
    for model in P.MODELS:
        for state in ("intact", "removal"):
            arm = [r for r in runs if r["model"] == model and r["state"] == state]
            print(model, state, "median peak sigma", round(float(np.median([r["peak_sigma"] for r in arm])), 3),
                  "median peak error", round(float(np.median([r["peak_error"] for r in arm])), 3))


if __name__ == "__main__":
    main()
