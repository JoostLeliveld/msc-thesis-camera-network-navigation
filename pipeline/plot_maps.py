#!/usr/bin/env python3
"""Review maps of the dataset, its camera coverage and the planning fields.

Writes logs/thesis/maps/:
  dataset.png      every position by role (colour) and capture pass (marker); repaired rings
  coverage.png     cameras that admit the robot at each working position (sensor gate)
  fields.png       one-sigma planning uncertainty (cm) per camera and network, M0/M1/M2
  dropout.png      per task: network field with its dropped camera removed, M0/M1/M2, and
                   the M2 loss (removal / intact)
The warehouse is drawn from the world file itself (every collision box, incl. the loose
objects), with the site boundary and the camera mounts.

    python3 pipeline/plot_maps.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/unav_common"), str(REPO / "world")]
from pipeline import dataset  # noqa: E402
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402
import warehouse_v2  # noqa: E402

FITS = REPO / "logs/thesis/fits"
OUT = REPO / "logs/thesis/maps"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
PROFILE = yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"][WORLD.name]
TASKS = yaml.safe_load((REPO / "pipeline/tasks.yaml").read_text())["tasks"][WORLD.name]
TEMPLATE = yaml.safe_load((REPO / "pipeline/execution_template.yaml").read_text())
ROLE_COLOUR = {"D_mu": "#2f8f5b", "D_R": "#1b6ca8", "D_dev": "#e8a33d", "final_audit": "#8e44ad"}
SOURCE_MARKER = {"v5": "o", "supplement": "s", "topup": "^"}
NO_SUPPORT_CM = 100.0


def draw_world(ax):
    site = next(r for r in PROFILE["known_2d_regions"] if r.get("type") == "site_boundary")
    ax.add_patch(plt.Rectangle((site["xmin"], site["ymin"]), site["xmax"] - site["xmin"],
                               site["ymax"] - site["ymin"], fill=False, ec="#3b6fb6", lw=1.2, ls="--"))
    for p in profile_collision_scene(str(WORLD), PROFILE).prisms:
        ax.add_patch(plt.Rectangle((p.xmin, p.ymin), p.xmax - p.xmin, p.ymax - p.ymin,
                                   fc="#c9c7bf", ec="#8f8d86", lw=0.4, zorder=1))
    for cam in warehouse_v2.build().cameras:
        ax.plot(cam.x, cam.y, marker="*", ms=11, color="#d62728", mec="black", mew=0.5, zorder=5)
        ax.annotate(cam.name, (cam.x, cam.y), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, weight="bold", zorder=6)
    ax.set_aspect("equal"); ax.set_xlim(-12.2, 12.2); ax.set_ylim(-10.2, 10.2)
    ax.set_xticks([]); ax.set_yticks([])


def position_table():
    rows = dataset.load_rows()
    table = {}
    for r in rows:
        src = r["capture_source"]
        key = r["position_key"]
        entry = table.setdefault(key, {"x": float(r["robot_x"]), "y": float(r["robot_y"]),
                                       "role": r["stratum"], "passes": set()})
        entry["passes"].add(src)
    for entry in table.values():
        p = entry["passes"]
        entry["source"] = "supplement" if "supplement" in p else "topup" if "topup" in p else "v5"
        entry["repaired"] = bool(p & {"repair", "repair2"})
    return table


def plot_dataset(table):
    fig, ax = plt.subplots(figsize=(12, 10))
    draw_world(ax)
    for role, colour in ROLE_COLOUR.items():
        for source, marker in SOURCE_MARKER.items():
            pts = [(e["x"], e["y"]) for e in table.values() if e["role"] == role and e["source"] == source]
            if pts:
                xy = np.asarray(pts)
                ax.scatter(xy[:, 0], xy[:, 1], s=16 if source == "v5" else 40, marker=marker, c=colour,
                           label=f"{role} / {source} ({len(pts)})", zorder=3, lw=0)
    rep = np.asarray([(e["x"], e["y"]) for e in table.values() if e["repaired"]])
    if len(rep):
        ax.scatter(rep[:, 0], rep[:, 1], s=30, facecolors="none", edgecolors="black", lw=0.5,
                   label=f"repaired poses ({len(rep)} positions)", zorder=4)
    counts = defaultdict(int)
    for e in table.values():
        counts[e["role"]] += 1
    ax.set_title(f"Reference dataset v8: {len(table)} positions  "
                 + "  ".join(f"{k} {counts[k]}" for k in ROLE_COLOUR), fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "dataset.png", dpi=150); plt.close(fig)


def plot_coverage(table):
    admitted = np.load(FITS / "gate_dataset/admitted.npz", allow_pickle=False)
    cams = defaultdict(set)
    for key, camera in zip(admitted["position_key"].astype(str), admitted["camera"].astype(str)):
        cams[key].add(camera)
    fig, ax = plt.subplots(figsize=(12, 10))
    draw_world(ax)
    working = [(e["x"], e["y"], len(cams.get(k, ()))) for k, e in table.items() if e["role"] != "final_audit"]
    xy = np.asarray(working)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=xy[:, 2], cmap="viridis", vmin=0, vmax=5, s=18, zorder=3)
    blind = xy[xy[:, 2] == 0]
    ax.scatter(blind[:, 0], blind[:, 1], s=60, facecolors="none", edgecolors="red", lw=1.2, zorder=4,
               label=f"no camera admits ({len(blind)} of {len(xy)} working positions)")
    audit = np.asarray([(e["x"], e["y"]) for e in table.values() if e["role"] == "final_audit"])
    ax.scatter(audit[:, 0], audit[:, 1], s=18, facecolors="none", edgecolors="#8e44ad", lw=0.8,
               zorder=3, label=f"final_audit, sealed ({len(audit)})")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="cameras admitting the robot (any heading)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1), fontsize=8, frameon=False)
    ax.set_title("Camera coverage under config/sensor_gate.yaml (working positions)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "coverage.png", dpi=150); plt.close(fig)
    return len(blind), len(xy)


def sigma_cm(precision):
    """One-sigma (geometric mean of the axes) of the covariance implied by a precision field."""
    det = np.linalg.det(precision)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = 100.0 * np.where(det > 0, det ** -0.25, np.inf)
    return np.ma.masked_where(~np.isfinite(sigma) | (sigma >= NO_SUPPORT_CM), sigma)


def load_field(model):
    with np.load(FITS / f"planning_precision/{model}_planning_precision.npz", allow_pickle=False) as a:
        return a["xs"], a["ys"], [str(c) for c in a["camera_ids"]], np.asarray(a["matched_precision_m2_inv"])


def show(ax, xs, ys, field, title):
    draw_world(ax)
    ax.pcolormesh(xs, ys, field, shading="nearest", norm=LogNorm(1.0, NO_SUPPORT_CM),
                  cmap="magma_r", zorder=2, alpha=0.85)
    ax.set_title(title, fontsize=8)


def plot_fields():
    fig, axes = plt.subplots(3, 6, figsize=(26, 12.5), layout="constrained")
    for row, model in enumerate(("m0", "m1", "m2")):
        xs, ys, cams, P = load_field(model)
        for col, cam in enumerate(cams):
            show(axes[row, col], xs, ys, sigma_cm(P[col]), f"{model.upper()} {cam}")
        show(axes[row, 5], xs, ys, sigma_cm(P.sum(axis=0)), f"{model.upper()} network (all cameras)")
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(1.0, NO_SUPPORT_CM), cmap="magma_r"), ax=axes,
                 shrink=0.6, label=f"one-sigma planning uncertainty (cm); blank = no support (>= {NO_SUPPORT_CM:.0f} cm)")
    fig.suptitle("Planning fields: inverse of the matched runtime covariance", fontsize=12)
    fig.savefig(OUT / "fields.png", dpi=110); plt.close(fig)


def plot_dropout():
    dropped = {name: t["condition_overrides"]["spatial_removal"]["removed_camera_id"]
               for name, t in TEMPLATE["tasks"].items()}
    tasks = [t for t in TASKS if t["name"] in dropped]
    fig, axes = plt.subplots(len(tasks), 4, figsize=(19, 4.2 * len(tasks)), layout="constrained")
    fields = {m: load_field(m) for m in ("m0", "m1", "m2")}
    for r, task in enumerate(tasks):
        cam = dropped[task["name"]]
        for c, model in enumerate(("m0", "m1", "m2")):
            xs, ys, cams, P = fields[model]
            keep = [i for i, name in enumerate(cams) if name != cam]
            ax = axes[r, c]
            show(ax, xs, ys, sigma_cm(P[keep].sum(axis=0)), f"{task['name'][9:]}\n{model.upper()} without {cam}")
            ax.plot(task["start"]["x"], task["start"]["y"], "go", ms=6, zorder=7)
            ax.plot(task["goal"]["x"], task["goal"]["y"], "bs", ms=6, zorder=7)
        xs, ys, cams, P = fields["m2"]
        keep = [i for i, name in enumerate(cams) if name != cam]
        loss = sigma_cm(P[keep].sum(axis=0)) / sigma_cm(P.sum(axis=0))
        ax = axes[r, 3]
        draw_world(ax)
        mesh = ax.pcolormesh(xs, ys, loss, shading="nearest", cmap="Reds", vmin=1, vmax=5, zorder=2, alpha=0.85)
        ax.set_title(f"M2 loss without {cam} (sigma ratio)", fontsize=8)
    fig.colorbar(mesh,  # noqa: F821 (set in the loop)
 ax=axes[:, 3], shrink=0.5, label="removal / intact one-sigma")
    fig.suptitle("Dropped-camera planning fields per task (green: start, blue: goal)", fontsize=12)
    fig.savefig(OUT / "dropout.png", dpi=100); plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table = position_table()
    plot_dataset(table)
    blind, working = plot_coverage(table)
    plot_fields()
    plot_dropout()
    print(json.dumps({"maps": str(OUT), "positions": len(table),
                      "working_positions_no_camera_admits": blind, "working_positions": working}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
