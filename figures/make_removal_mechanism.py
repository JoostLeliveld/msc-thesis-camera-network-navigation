#!/usr/bin/env python3
"""The mechanism in one task: what each model believes it loses when a camera goes, and
what the robot then does.

Rows: all cameras / dropout of the task camera. Columns: global, per-camera, spatial model.
Background: the model's own planning field, half the trace of its summed per-camera
precision (m^-2), on the colour scale of the methodology's planner-field figure; where the
dropped camera was the only support, the field falls to zero.
Lines: the route the model planned (thick) and the three executed true paths (thin);
x marks where a footprint first left the driveable region, o where a failed run stopped without leaving it.

    python3 figures/make_removal_mechanism.py [TASK]
"""
from __future__ import annotations

import csv
import sys

import numpy as np

import paper as P
from pipeline.score_collisions import read_poses

TASK = sys.argv[1] if len(sys.argv) > 1 else "thesis10_camera_c_inner_warehouse_detour"


def runs():
    with (P.ANALYSIS / "runs.csv").open(newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["task"] == TASK]


def main():
    task = next(t for t in P.tasks() if t["name"] == TASK)
    rows = runs()
    vmax = P.information_vmax()
    fig, axes = P.plt.subplots(2, 3, figsize=(P.TEXT, 3.95), gridspec_kw=dict(wspace=0.12, hspace=0.12))
    mesh = None
    for i, state in enumerate(("intact", "removal")):
        removed = task["removed"] if state == "removal" else None
        for j, model in enumerate(P.MODELS):
            ax = axes[i, j]
            xs, ys, info = P.network_information(model, removed)
            mesh = P.draw_information(ax, xs, ys, info, vmax)
            P.draw_map(ax, removed=removed, camera_labels=(j == 0))
            colour = P.MODEL_COLOUR[model]
            for r in rows:
                if r["model"] != model or r["state"] != state:
                    continue
                poses = np.array([p[1:3] for p in read_poses(P.REPO / r["run_dir"] / "ground_truth_pose.csv")])
                ax.plot(poses[:, 0], poses[:, 1], color=P.INK, lw=1.7, zorder=5, alpha=0.9)
                ax.plot(poses[:, 0], poses[:, 1], color=colour, lw=0.7, zorder=6)
                if r["collision"] == "1":
                    ax.plot(float(r["collision_x"]), float(r["collision_y"]), "x", color=P.COLLISION,
                            ms=6, mew=1.4, zorder=10)
                elif r["success"] != "1":
                    ax.plot(poses[-1, 0], poses[-1, 1], "o", mfc="none", mec=P.STUCK, ms=5, mew=1.2, zorder=10)
            wins = sum(r["success"] == "1" for r in rows if r["model"] == model and r["state"] == state)
            ax.text(0.99, 0.995, f"{wins}/3 succeeded", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=P.INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=12)
            route = P.planned_route(TASK, f"{model}_{state}")
            ax.plot(route[:, 0], route[:, 1], color="white", lw=3.4, alpha=0.8, zorder=3.9,
                    solid_capstyle="round")
            ax.plot(route[:, 0], route[:, 1], color=colour, lw=2.2, alpha=0.6, zorder=4,
                    solid_capstyle="round")
            ax.plot(task["start"]["x"], task["start"]["y"], "o", ms=4, color=P.INK, zorder=11)
            ax.plot(task["goal"]["x"], task["goal"]["y"], "*", ms=8, color="white", mec=P.INK, mew=0.6, zorder=11)
            if i == 0:
                ax.set_title(P.MODEL_LABEL[model], color=colour, fontweight="bold", pad=2)
        axes[i, 0].text(-0.03, 0.5, "all cameras" if state == "intact" else f"camera {task['removed'][-1]} dropout",
                        transform=axes[i, 0].transAxes, rotation=90, ha="right", va="center", fontsize=8)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.72, pad=0.01, aspect=28, extend="max")
    cbar.set_label(r"$\frac{1}{2}\mathrm{tr}\sum_i\Lambda_{m,i}(p)$ (m$^{-2}$)")
    cbar.outline.set_linewidth(0.4)
    # camera letter plus task name: the letter alone collides for the two camera-E tasks
    P.save(fig, f"removal_mechanism_{TASK.removeprefix('thesis10_camera_')}")


if __name__ == "__main__":
    main()
