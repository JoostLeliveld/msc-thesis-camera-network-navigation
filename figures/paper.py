"""Shared look for the manuscript figures: IEEE sizes, one colour per model, one map.

Every figure reads from `logs/thesis/` only. The warehouse is drawn from the world file
itself (every collision box, including the five loose objects), the same scene the
collision score uses, so a map never shows free space the robot does not have.

| colour | meaning |
|---|---|
| blue `#0072B2` | global covariance model (R0 / M0) |
| orange `#E69F00` | per-camera model (R1 / M1) |
| green `#009E73` | spatial model (R2 / M2) |
| grey | geometric projection baseline (R_proj) and other baselines |
| vermillion `#D55E00` | a collision (footprint left the driveable region) |
| reddish purple `#CC79A7` | a failed run that stayed inside the driveable region |

Figures are written to `logs/thesis/figures/` (PDF for the manuscript, PNG for review).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/unav_common"), str(REPO / "world")]
THESIS = REPO / "logs/thesis"
FINAL = THESIS / "final_campaign"
ANALYSIS = FINAL / "analysis"
ROUTES = FINAL / "routes"
OUT = THESIS / "figures"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"

COLUMN, TEXT = 3.5, 7.16          # IEEE column and text width (inches)
MODELS = ("global", "per_camera", "spatial")
MODEL_KEY = {"global": "m0", "per_camera": "m1", "spatial": "m2"}
MODEL_LABEL = {"global": "Global", "per_camera": "Per-camera", "spatial": "Spatial",
               "rproj": "Geometric", "equal": "Equal weights", "best_spatial_single": "Best single"}
MODEL_COLOUR = {"global": "#0072B2", "per_camera": "#E69F00", "spatial": "#009E73",
                "rproj": "#7f7f7f", "equal": "#b0b0b0", "best_spatial_single": "#595959"}
COLLISION, STUCK, SHORT = "#D55E00", "#CC79A7", "#a6a6a6"
INK, MUTED, RACK, RACK_EDGE = "#1a1a1a", "#6b6b6b", "#e4e2dc", "#bdbab2"
from matplotlib.colors import ListedColormap  # noqa: E402
# Reversed viridis, so low uncertainty (high information) is yellow as in the methodology's
# planner-field figure; routes carry a dark halo to stay readable on its yellow end.
SIGMA_CMAP = ListedColormap(plt.get_cmap("viridis_r")(np.linspace(0.0, 1.0, 256)), name="sigma")
SIGMA_RANGE_CM = (1.5, 60.0)
NO_SUPPORT_CM = 100.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 150,
    "hatch.color": "#d9a38a", "hatch.linewidth": 0.45,
})


def cameras():
    import warehouse_v2
    return {c.name: c for c in warehouse_v2.build().cameras}


def _scene():
    from unav_common.occlusion_geometry import profile_collision_scene
    profile = yaml.safe_load((REPO / "src/experiments/config/world_profiles.yaml").read_text())["worlds"][WORLD.name]
    site = next(r for r in profile["known_2d_regions"] if r.get("type") == "site_boundary")
    return profile_collision_scene(str(WORLD), profile).prisms, site


_SCENE = None


def draw_map(ax, *, removed: str | None = None, camera_labels: bool = True, cams: bool = True):
    """Site boundary, every collision box, camera mounts; a removed camera is hollow."""
    global _SCENE
    if _SCENE is None:
        _SCENE = _scene()
    prisms, site = _SCENE
    ax.add_patch(Rectangle((site["xmin"], site["ymin"]), site["xmax"] - site["xmin"],
                           site["ymax"] - site["ymin"], fill=False, ec=MUTED, lw=0.6, zorder=1))
    for p in prisms:
        ax.add_patch(Rectangle((p.xmin, p.ymin), p.xmax - p.xmin, p.ymax - p.ymin,
                               fc=RACK, ec=RACK_EDGE, lw=0.3, zorder=3))
    if cams:
        for name, c in cameras().items():
            gone = removed is not None and removed.endswith(name)
            # A removed camera is hollow and grey: never a cross, which means a collision.
            ax.plot(c.x, c.y, marker="^", ms=4.5, color="white" if gone else INK,
                    mec=MUTED if gone else INK, mew=0.8, zorder=8)
            if camera_labels or gone:
                # Labels sit outside the site so they never cover a route.
                dy = -0.75 if c.y < 0 else 0.75
                ax.text(c.x, c.y + dy, f"{name} off" if gone else name, fontsize=6.5,
                        ha="center", va="center",
                        color=MUTED if gone else INK, fontweight="bold",
                        fontstyle="italic" if gone else "normal", zorder=9)
    ax.set_xlim(site["xmin"] - 0.5, site["xmax"] + 0.5)
    ax.set_ylim(site["ymin"] - 1.2, site["ymax"] + 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def precision_field(model: str):
    with np.load(THESIS / f"fits/planning_precision/{MODEL_KEY[model]}_planning_precision.npz",
                 allow_pickle=False) as a:
        return a["xs"], a["ys"], [str(c) for c in a["camera_ids"]], np.asarray(a["matched_precision_m2_inv"])


def sigma_cm(precision_sum):
    """One-sigma (geometric mean of the axes) implied by a summed precision, in cm."""
    det = np.linalg.det(precision_sum)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = 100.0 * np.where(det > 0, det ** -0.25, np.inf)
    return np.ma.masked_where(~np.isfinite(sigma) | (sigma >= NO_SUPPORT_CM), sigma)


def network_sigma(model: str, removed: str | None = None):
    xs, ys, cams, P = precision_field(model)
    keep = [i for i, c in enumerate(cams) if c != removed]
    return xs, ys, sigma_cm(P[keep].sum(axis=0))


def network_information(model: str, removed: str | None = None):
    """Half the trace of the summed camera precision, the planner-field quantity (m^-2)."""
    xs, ys, cams, P = precision_field(model)
    keep = [i for i, c in enumerate(cams) if c != removed]
    return xs, ys, 0.5 * np.trace(P[keep].sum(axis=0), axis1=-2, axis2=-1)


def information_vmax() -> float:
    """The planner-field figure's cap: the 90th percentile of the intact spatial field."""
    return float(np.percentile(network_information("spatial")[2], 90))


def draw_information(ax, xs, ys, values, vmax):
    return ax.pcolormesh(xs, ys, values, shading="nearest", cmap="viridis", vmin=0.0, vmax=vmax,
                         zorder=2, rasterized=True)


def draw_sigma(ax, xs, ys, sigma, norm):
    """The field over a hatched ground: hatch shows through wherever no camera supports it."""
    ax.add_patch(Rectangle((xs[0], ys[0]), xs[-1] - xs[0], ys[-1] - ys[0], fc="white",
                           ec="#d9a38a", hatch="////", lw=0, zorder=1.5))
    return ax.pcolormesh(xs, ys, sigma, shading="nearest", cmap=SIGMA_CMAP, norm=norm,
                         zorder=2, rasterized=True)


# Tasks B and C swap their removed camera and moved their goals; their runs and routes live
# here and replace the campaign's for those tasks (as in pipeline/analyze_campaign.py).
BC_SWAP = THESIS / "revisions/bc_dropout_swap"


def tasks():
    cfg = yaml.safe_load((FINAL / "campaign_configs/campaign_seed91500.yaml").read_text())
    spec = {t["name"]: t for t in yaml.safe_load((REPO / "pipeline/tasks.yaml").read_text())["tasks"][WORLD.name]}
    out = []
    for name, t in cfg["tasks"].items():
        out.append({"name": name, "removed": t["condition_overrides"]["spatial_removal"]["removed_camera_id"],
                    "start": spec[name]["start"], "goal": spec[name]["goal"]})
    return out


TASK_LABEL = {
    "thesis10_camera_a_western_dock_detour": "A: western dock",
    "thesis10_camera_b_cross_warehouse_detour": "B: cross-warehouse",
    "thesis10_camera_c_inner_warehouse_detour": "C: inner warehouse",
    "thesis10_camera_e_eastern_detour": "E-east",
    "thesis10_camera_e_long_cross_warehouse_detour": "E-long",
}


def planned_route(task: str, condition: str) -> np.ndarray:
    root = ROUTES
    result = json.loads((root / f"{task}/manifest.json").read_text())["results"][condition]
    return np.asarray(json.loads((root / f"{task}/{result['preselected_route']['path']}").read_text()))


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(OUT / f"{stem}.pdf")
