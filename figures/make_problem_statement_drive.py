#!/usr/bin/env python3
"""The problem-statement drive: raw camera errors in the camera-ray frame.

Uses the same intact campaign run as make_navigation_single_run.py, so the
results figure shows the corrected counterpart of exactly this drive.  Only the
raw projection is plotted; no correction or covariance enters this figure.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from make_navigation_single_run import (  # noqa: E402
    INK, PAPER_FIGURES, draw_raw, load)
from style import CAM_COLOUR, draw_warehouse, layout  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="thesis10_camera_a_western_dock_detour")
    parser.add_argument("--seed", default="seed91500")
    parser.add_argument("--condition", default="spatial_intact")
    arguments = parser.parse_args()

    run = load(arguments.task, arguments.condition, arguments.seed)
    fusion, experiment, waypoints = run["fusion"], run["experiment"], run["waypoints"]
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5,
                         "xtick.labelsize": 5.8, "ytick.labelsize": 5.8})

    fig = plt.figure(figsize=(3.5, 4.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(1.9, 1.0, 1.0))

    ax_map = fig.add_subplot(grid[0])
    draw_warehouse(ax_map, layout(), show_cameras=True, camera_labels=True, rack_alpha=0.82)
    ax_map.plot(waypoints["x"], waypoints["y"], color="#555b63", lw=0.9, ls="--",
                alpha=0.75, zorder=14)
    ax_map.plot(experiment["gt_x"], experiment["gt_y"], color=INK, lw=1.6, zorder=15)
    for camera, group in fusion.groupby("camera"):
        ax_map.scatter(group["raw_obs_x"], group["raw_obs_y"], s=4,
                       color=CAM_COLOUR[str(camera)], lw=0, alpha=0.6, zorder=16)
    ax_map.plot(experiment["gt_x"].iloc[0], experiment["gt_y"].iloc[0], "o", ms=4.5,
                mfc="white", mec=INK, zorder=20)
    ax_map.plot(waypoints["x"].iloc[-1], waypoints["y"].iloc[-1], "*", ms=8,
                mfc="white", mec=INK, zorder=20)
    ax_map.set(xlim=(-11.8, 11.8), ylim=(-9.75, 9.75), aspect="equal",
               xticks=[-10, -5, 0, 5, 10], yticks=[-5, 0, 5],
               xlabel="x [m]", ylabel="y [m]")

    first = None
    for row, (column, label) in enumerate(
            (("raw_par", "along-ray error [cm]"), ("raw_perp", "across-ray error [cm]")),
            start=1):
        ax = fig.add_subplot(grid[row], sharex=first)
        first = first or ax
        draw_raw(ax, fusion, column, show_camera_means=False)
        limit = 1.08 * max(abs(fusion[column].quantile([0.005, 0.995]))) * 100
        ax.set_ylim(-limit, limit)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e2e2e2", lw=0.45)
        if row == 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("time after first command [s]")

    handles = [Line2D([], [], marker="o", ls="none", color=CAM_COLOUR[c], ms=4,
                      label=f"camera {c}")
               for c in sorted(fusion["camera"].astype(str).unique())]
    handles += [Line2D([], [], color="#555b63", ls="--", lw=0.9, label="planned path"),
                Line2D([], [], color=INK, lw=1.4, label="executed trajectory")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=5.6, bbox_to_anchor=(0.5, -0.075))

    for suffix in ("pdf", "png"):
        path = PAPER_FIGURES / f"problem_statement_drive.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
