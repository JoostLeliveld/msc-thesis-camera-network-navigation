#!/usr/bin/env python3
"""All five tasks under dropout of their camera: where the information goes, what each model
planned, and what the robot did.

Background: the spatial model's planning field under dropout of the task camera (one-sigma cm,
hatched where no camera supports a position). Thick translucent lines: the planned route of
each model under removal (the per-camera route is drawn only where it differs from the
global one). Thin lines: the three executed true paths per model. Glyphs: x the footprint
first left the driveable region, o a failed run stopped without leaving it.

    python3 figures/make_campaign_routes.py
"""
from __future__ import annotations

import csv

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import paper as P
from pipeline.score_collisions import read_poses


def main():
    with (P.ANALYSIS / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    norm = LogNorm(*P.SIGMA_RANGE_CM)
    fig, axes = P.plt.subplots(2, 3, figsize=(P.TEXT, 3.75), gridspec_kw=dict(wspace=0.04, hspace=0.2))
    mesh = None
    for ax, task in zip(axes.flat, P.tasks()):
        name, removed = task["name"], task["removed"]
        xs, ys, sigma = P.network_sigma("spatial", removed)
        mesh = P.draw_sigma(ax, xs, ys, sigma, norm)
        P.draw_map(ax, removed=removed, camera_labels=False)
        routes = {m: P.planned_route(name, f"{m}_removal") for m in P.MODELS}
        for model in P.MODELS:
            if model == "per_camera" and routes["per_camera"].shape == routes["global"].shape \
                    and np.allclose(routes["per_camera"], routes["global"]):
                continue
            r = routes[model]
            ax.plot(r[:, 0], r[:, 1], color=P.MODEL_COLOUR[model], lw=3.2, alpha=0.3, zorder=4,
                    solid_capstyle="round")
        counts = []
        for model in P.MODELS:
            runs = [r for r in rows if r["task"] == name and r["model"] == model and r["state"] == "removal"]
            counts.append(sum(r["success"] == "1" for r in runs))
            for r in runs:
                poses = np.array([p[1:3] for p in read_poses(P.REPO / r["run_dir"] / "ground_truth_pose.csv")])
                ax.plot(poses[:, 0], poses[:, 1], color=P.MODEL_COLOUR[model], lw=0.6, zorder=6, alpha=0.95)
                if r["collision"] == "1":
                    ax.plot(float(r["collision_x"]), float(r["collision_y"]), "x", color=P.COLLISION, ms=5.5, mew=1.3, zorder=10)
                elif r["success"] != "1":
                    ax.plot(poses[-1, 0], poses[-1, 1], "o", mfc="none", mec=P.STUCK, ms=4.5, mew=1.1, zorder=10)
        ax.plot(task["start"]["x"], task["start"]["y"], "o", ms=3.5, color=P.INK, zorder=11)
        ax.plot(task["goal"]["x"], task["goal"]["y"], "*", ms=7.5, color="white", mec=P.INK, mew=0.6, zorder=11)
        ax.set_title(f"{P.TASK_LABEL[name]}  (camera {removed[-1]} dropout)", pad=2, fontsize=7.5)
        for k, (m, c) in enumerate(zip(P.MODELS, counts)):
            ax.text(0.17 + 0.33 * k, -0.02, f"{P.MODEL_LABEL[m]} {c}/3", transform=ax.transAxes, ha="center",
                    va="top", fontsize=6.3, color=P.MODEL_COLOUR[m], fontweight="bold")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = [Line2D([], [], color=P.MODEL_COLOUR[m], lw=3.2, alpha=0.35) for m in P.MODELS] + \
              [Line2D([], [], color=P.INK, lw=0.6),
               Line2D([], [], marker="x", ls="none", color=P.COLLISION, ms=5.5, mew=1.3),
               Line2D([], [], marker="o", ls="none", mfc="none", mec=P.STUCK, ms=4.5, mew=1.1),
               Line2D([], [], marker="^", ls="none", mfc="white", mec=P.MUTED, ms=5, mew=0.8)]
    labels = [f"{P.MODEL_LABEL[m]} planned route" for m in P.MODELS] + \
             ["executed true paths (3 seeds)", "failed, left the driveable region", "failed, stayed inside", "dropped camera"]
    legend_ax.legend(handles, labels, loc="upper left", fontsize=6.8, handlelength=2.2, borderaxespad=0.2)
    cax = legend_ax.inset_axes([0.05, 0.08, 0.85, 0.07])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_ticks([2, 5, 10, 20, 50]); cbar.set_ticklabels(["2", "5", "10", "20", "50"]); cbar.minorticks_off()
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("spatial-model uncertainty under dropout,\none sigma (cm); hatched: no support",
                   fontsize=6.3, labelpad=2)
    cbar.ax.tick_params(labelsize=6)
    P.save(fig, "campaign_routes")


if __name__ == "__main__":
    main()
