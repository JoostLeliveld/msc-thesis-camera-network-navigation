#!/usr/bin/env python3
"""Inspect the 30 solved offline routes used by the final campaign.

One row per task, one column per covariance model. Each panel shows the intact route
(solid) and the dropout route (dashed) over the network information of that model with
the dropped camera removed, and crosses out the dropped camera. A summary table with the
selected seed route, length, costs and minimum predicted clearance is printed and saved.

    python3 pipeline/plot_routes.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/unav_common"))
from unav_common.occlusion_geometry import profile_collision_scene  # noqa: E402

V8 = REPO / "logs/thesis"
ROUTES = V8 / "routes"
MODELS = (("global", "m0", "R$_0$ global"), ("per_camera", "m1", "R$_1$ per-camera"),
          ("spatial", "m2", "R$_2$ spatial"))
CAMS = {"camera_A": (-11.45, -9.45), "camera_B": (-1.5, -9.72), "camera_C": (-6.95, 9.45),
        "camera_D": (11.45, 7.2), "camera_E": (11.45, -9.45)}


def network_information(model_key: str, active: list[str]):
    z = np.load(V8 / f"fits/planning_precision/{model_key}_planning_precision.npz")
    ids = [str(c) for c in z["camera_ids"]]
    info = sum(np.trace(z["matched_precision_m2_inv"][ids.index(c)], axis1=-2, axis2=-1)
               for c in active)
    return z["xs"], z["ys"], info


def main() -> int:
    campaign = yaml.safe_load((V8 / "campaign_configs/route_planning_campaign.yaml").read_text())
    tasks = list(campaign["tasks"])
    scene = profile_collision_scene(
        str(REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"), yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"]["warehouse_v2.world.sdf"])
    fig, axes = plt.subplots(len(tasks), 3, figsize=(15, 4.1 * len(tasks)), constrained_layout=True)
    rows = []
    for t, task in enumerate(tasks):
        manifest = json.loads((ROUTES / task / "manifest.json").read_text())
        override = campaign["tasks"][task]["condition_overrides"]["spatial_removal"]
        dropped = override["removed_camera_id"]
        active_drop = override["camera_network_active_camera_ids"].split(",")
        for m, (prefix, key, title) in enumerate(MODELS):
            ax = axes[t, m]
            xs, ys, info = network_information(key, active_drop)
            ax.imshow(np.maximum(info, 1e-3), origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                      cmap="viridis", norm=LogNorm(vmin=1e0, vmax=np.percentile(info, 99)), alpha=0.75)
            for pr in scene.prisms:
                if "wall" not in pr.name:
                    ax.add_patch(Rectangle((pr.xmin, pr.ymin), pr.xmax - pr.xmin, pr.ymax - pr.ymin,
                                           fc="#d9c9a8", ec="#6e5a33", lw=0.4, zorder=2))
            lengths = {}
            for state, style, colour in (("intact", "-", "white"), ("removal", "--", "#ff5a36")):
                condition = f"{prefix}_{state}"
                result = manifest["results"][condition]
                route = np.array(json.loads((ROUTES / task / result["preselected_route"]["path"]).read_text()))
                ax.plot(route[:, 0], route[:, 1], style, color=colour, lw=2.4, zorder=4,
                        label="intact" if state == "intact" else f"{dropped[-1]} dropped")
                length = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
                lengths[state] = (route, length)
                rows.append({"task": task.replace("thesis10_", ""), "model": prefix, "state": state,
                             "route": result.get("selected_source", "").replace("seed:route:", ""),
                             "length_m": round(length, 2), "total_cost": round(result["total_cost"], 3),
                             "risk": round(result["risk_cost"], 3), "ambiguity": round(result["ambiguity_cost"], 3),
                             "no_go": round(result["obstacle_cost"], 4),
                             "min_pred_clearance_m": round(result["minimum_predicted_clearance_m"], 3)})
            changed = not (lengths["intact"][0].shape == lengths["removal"][0].shape
                           and np.allclose(lengths["intact"][0], lengths["removal"][0]))
            rows[-1]["changed"] = rows[-2]["changed"] = changed
            ax.plot(*route[0], "o", color="white", mec="k", ms=8, zorder=5)
            ax.plot(*route[-1], "*", color="gold", mec="k", ms=14, zorder=5)
            for cam, (x, y) in CAMS.items():
                ax.plot(x, y, "s", color="#1f3b73", ms=8, zorder=5)
                ax.annotate(cam[-1], (x, y), xytext=(4, 4), textcoords="offset points", color="white",
                            weight="bold", zorder=6)
                if cam == dropped:
                    ax.plot(x, y, "x", color="#ff2020", ms=16, mew=3, zorder=6)
            ax.set_xlim(-12, 12); ax.set_ylim(-10, 10); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            tag = "CHANGED" if changed else "same route"
            ax.set_title(f"{task.replace('thesis10_', '')}\n{title}: {tag}  "
                         f"({lengths['intact'][1]:.1f} m / {lengths['removal'][1]:.1f} m)", fontsize=10)
            if t == 0 and m == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Final offline routes: intact (solid) vs camera removal (dashed), over the camera-network "
                 "information of each model", fontsize=13)
    out = ROUTES / "routes_overview.png"
    fig.savefig(out, dpi=110)
    (ROUTES / "routes_summary.json").write_text(json.dumps(rows, indent=1))
    print(f"{'task':34}{'model':11}{'state':8}{'route':22}{'len':>7}{'cost':>8}{'risk':>7}{'amb':>7}{'nogo':>7}{'clear':>7} changed")
    for r in rows:
        print(f"{r['task']:34}{r['model']:11}{r['state']:8}{r['route']:22}{r['length_m']:7.2f}{r['total_cost']:8.3f}"
              f"{r['risk']:7.3f}{r['ambiguity']:7.3f}{r['no_go']:7.4f}{r['min_pred_clearance_m']:7.3f} {r['changed']}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
