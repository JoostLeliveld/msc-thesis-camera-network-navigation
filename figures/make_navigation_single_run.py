#!/usr/bin/env python3
"""Follow one intact run through the deployed chain, in the problem figure's terms.

The problem statement plots the raw along-ray and across-ray residuals of one
drive.  This figure takes one intact campaign run and shows the same two
components before correction (left column, as in the problem figure) and after
correction with the runtime covariance's predicted two-sigma band (right
column), plus the fused estimate against its own predicted spread.  The
positions driven here were available to the covariance fit, so this shows the
deployed model calibrated in use, not an out-of-sample test.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from style import CAM_COLOUR, draw_warehouse, layout  # noqa: E402

PAPER_FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
CAMPAIGN = Path(__import__("os").environ.get(
    "THESIS_CAMPAIGN_ROOT", ROOT / "logs/thesis/final_campaign/campaign"))
INK = "#222831"
BELIEF = "#2a78d6"
GAP = 0.5  # s; a longer pause in one camera's stream breaks its band


def run_directory(task: str, condition: str, seed: str) -> Path:
    matches = sorted(glob.glob(str(
        CAMPAIGN / seed / task / condition / seed / "attempts/*/experiment_*/")))
    if len(matches) != 1:
        raise ValueError(f"{task}/{condition}/{seed}: expected one run, found {len(matches)}")
    return Path(matches[0])


def load(task: str, condition: str, seed: str) -> dict:
    directory = run_directory(task, condition, seed)
    experiment = pd.read_csv(directory / "experiment.csv")
    step = np.hypot(experiment["state_x"].diff().fillna(0.0),
                    experiment["state_y"].diff().fillna(0.0))
    moving = step.cumsum() > 0.05
    if not moving.any():
        raise ValueError(f"{directory}: the robot never moved")
    # t = 0 is the first commanded motion, not the first logged sample.
    origin = float(experiment["stamp"][moving].iloc[0])
    experiment["t"] = experiment["stamp"] - origin
    experiment = experiment[experiment["t"] >= -1.0].copy()

    fusion = pd.read_csv(directory / "fusion_observations.csv")
    fusion = fusion[fusion["used"].astype(bool)].copy()
    fusion["t"] = fusion["stamp"] - origin
    fusion = fusion[fusion["t"] >= -1.0].copy()

    # The camera-ray frame of the problem figure: parallel points from the
    # camera through the raw measurement, perpendicular is its rotation.
    mount = {c.name: np.array([c.x, c.y]) for c in layout().cameras}
    camera = np.stack([mount[str(name)] for name in fusion["camera"]])
    raw = fusion[["raw_obs_x", "raw_obs_y"]].to_numpy()
    truth = fusion[["gt_x_at_obs", "gt_y_at_obs"]].to_numpy()
    corrected = fusion[["obs_x", "obs_y"]].to_numpy()
    ray = raw - camera
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    across = np.column_stack([-ray[:, 1], ray[:, 0]])
    cov = np.stack([
        np.stack([fusion["obs_cov_xx"], fusion["obs_cov_xy"]], axis=-1),
        np.stack([fusion["obs_cov_xy"], fusion["obs_cov_yy"]], axis=-1)], axis=-2)
    for name, basis in (("par", ray), ("perp", across)):
        fusion[f"raw_{name}"] = np.einsum("ij,ij->i", raw - truth, basis)
        fusion[f"corr_{name}"] = np.einsum("ij,ij->i", corrected - truth, basis)
        # Predicted one-sigma of the runtime covariance along this direction.
        fusion[f"sd_{name}"] = np.sqrt(np.einsum("ni,nij,nj->n", basis, cov, basis))

    batches = fusion.groupby("source_batch_id").agg(
        t=("t", "mean"), n=("camera", "size"),
        fused_x=("fused_x", "first"), fused_y=("fused_y", "first"),
        gx=("gt_x_at_fused", "first"), gy=("gt_y_at_fused", "first"),
        cxx=("fused_cov_xx", "first"), cyy=("fused_cov_yy", "first"),
    ).sort_values("t")
    batches["error"] = np.hypot(batches["fused_x"] - batches["gx"],
                                batches["fused_y"] - batches["gy"])
    batches["sd"] = np.sqrt(0.5 * (batches["cxx"] + batches["cyy"]))

    return {"experiment": experiment, "fusion": fusion, "batches": batches,
            "waypoints": pd.read_csv(directory / "global_waypoints.csv")}


def broken(t: np.ndarray, *values: np.ndarray):
    """Insert NaN where a camera's stream pauses, so bands do not bridge gaps."""
    cut = np.where(np.diff(t) > GAP)[0] + 1
    return [np.insert(v.astype(float), cut, np.nan) for v in (t, *values)]


def draw_map(ax: plt.Axes, run: dict) -> None:
    draw_warehouse(ax, layout(), show_cameras=True, camera_labels=True, rack_alpha=0.82)
    waypoints, experiment = run["waypoints"], run["experiment"]
    ax.plot(waypoints["x"], waypoints["y"], color="#555b63", lw=0.9, ls="--",
            alpha=0.75, zorder=14)
    ax.plot(experiment["gt_x"], experiment["gt_y"], color=INK, lw=1.6, zorder=15)
    for camera, group in run["fusion"].groupby("camera"):
        ax.scatter(group["gt_x_at_obs"], group["gt_y_at_obs"], s=4,
                   color=CAM_COLOUR[str(camera)], lw=0, alpha=0.6, zorder=16)
    ax.plot(experiment["gt_x"].iloc[0], experiment["gt_y"].iloc[0], "o", ms=4.5,
            mfc="white", mec=INK, zorder=20)
    ax.plot(waypoints["x"].iloc[-1], waypoints["y"].iloc[-1], "*", ms=8,
            mfc="white", mec=INK, zorder=20)
    ax.set(xlim=(-11.8, 11.8), ylim=(-9.75, 9.75), aspect="equal")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-5, 0, 5])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) Executed route and admitted observations", loc="left")


def draw_raw(
        ax: plt.Axes, fusion: pd.DataFrame, column: str, *, show_camera_means: bool = True
) -> None:
    ax.axhline(0, color=INK, lw=0.7)
    for camera, group in fusion.groupby("camera"):
        colour = CAM_COLOUR[str(camera)]
        ax.scatter(group["t"], 100 * group[column], s=3, color=colour, lw=0, alpha=0.65)
        if show_camera_means:
            ax.axhline(100 * group[column].mean(), color=colour, lw=0.65, ls="--", alpha=0.8)


def draw_corrected(ax: plt.Axes, fusion: pd.DataFrame, name: str) -> None:
    ax.axhline(0, color=INK, lw=0.7)
    for camera, group in fusion.groupby("camera"):
        colour = CAM_COLOUR[str(camera)]
        group = group.sort_values("t")
        t, sd = broken(group["t"].to_numpy(), 100 * group[f"sd_{name}"].to_numpy())
        ax.fill_between(t, -2 * sd, 2 * sd, color=colour, alpha=0.13, lw=0)
        ax.plot(t, 2 * sd, color=colour, lw=0.6, alpha=0.8)
        ax.plot(t, -2 * sd, color=colour, lw=0.6, alpha=0.8)
        ax.scatter(group["t"], 100 * group[f"corr_{name}"], s=3, color=colour,
                   lw=0, alpha=0.85, zorder=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="thesis10_camera_a_western_dock_detour")
    parser.add_argument("--seed", default="seed91500")
    parser.add_argument("--condition", default="spatial_intact")
    arguments = parser.parse_args()

    run = load(arguments.task, arguments.condition, arguments.seed)
    fusion, batches = run["fusion"], run["batches"]
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.5,
                         "axes.titleweight": "bold", "axes.labelsize": 6.5,
                         "xtick.labelsize": 5.8, "ytick.labelsize": 5.8})

    fig = plt.figure(figsize=(7.16, 2.75), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.25))
    ax_map = fig.add_subplot(grid[:, 0])
    draw_map(ax_map, run)
    ax_map.set_title("(a) Executed route and admitted observations", loc="left")

    rows = (("par", "along-ray error [cm]"), ("perp", "across-ray error [cm]"))
    first = None
    for row, (name, label) in enumerate(rows):
        ax = fig.add_subplot(grid[row, 1], sharex=first)
        first = first or ax
        draw_corrected(ax, fusion, name)
        ax.set_ylim(-12, 12)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e2e2e2", lw=0.45)
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_title(r"(b) Corrected residuals with predicted $\pm2\sigma$", loc="left")
        else:
            ax.set_xlabel("time after first command [s]")

    handles = [Line2D([], [], marker="o", ls="none", color=CAM_COLOUR[c], ms=4,
                      label=f"camera {c}")
               for c in sorted(fusion["camera"].astype(str).unique())]
    handles += [Line2D([], [], color="#555b63", ls="--", lw=0.9, label="planned path"),
                Line2D([], [], color=INK, lw=1.4, label="executed trajectory")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=6, bbox_to_anchor=(0.5, -0.09))

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = PAPER_FIGURES / f"navigation_single_run.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
