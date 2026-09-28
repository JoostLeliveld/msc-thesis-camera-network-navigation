"""The deck's visual vocabulary: one warehouse, one set of colours, one meaning each.

Every figure is read by someone who has not seen the code, often from the back of a room.
So: heavy lines, few labels, and the same colour meaning the same thing on every slide.

| colour | meaning |
|---|---|
| blue | the robot, its onboard localization, its planned path |
| green | a useful camera sighting, good external support |
| orange | an outage, poor support, growing uncertainty |
| violet | the previous method |
| deep blue ramp | chance of a usable sighting -- a magnitude, so one hue light to dark |
| orange hatching | places no camera ever helped -- texture as well as colour |

Cameras keep their own five colours throughout, so a camera means the same thing on the
map, in the handover strip and in any per-camera plot.  The ramp is a single hue because it
encodes a magnitude, and the dead cells are hatched so "bad" never rests on colour alone.
"""
from __future__ import annotations
import collections, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
for p in ("world", "src/experiments", "src/unav_common"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURF = "#fcfcfb"
ROBOT = "#2a78d6"      # robot, onboard localization, planned path
GOOD = "#1baf7a"       # a useful camera observation
BAD = "#eb6834"        # outage, poor support, uncertainty growth
OLD = "#4a3aa7"        # the previous method
RACK = "#dcdbd4"
RACK_EDGE = "#b8b7ae"
SUPPORT = LinearSegmentedColormap.from_list("support", ["#f4f7fa", "#bcd6ef", "#5c9bd8", "#1f5c9e"])
CAM_COLOUR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#4a3aa7", "E": "#e34948"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#d5d4cf", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.grid": False, "axes.axisbelow": True,
})


def layout():
    import warehouse_v2 as W
    return W.build()


def driveable():
    import route_tasks as RT
    return RT.driveable()


def draw_warehouse(ax, lay, show_cameras=True, camera_labels=True, rack_alpha=1.0):
    """The building: walls, racks and block stacks, and the five camera mounts."""
    ax.add_patch(plt.Rectangle((-12, -10), 24, 20, facecolor="none",
                               edgecolor=MUTED, lw=2.4, zorder=1))
    for z in lay.zones:
        ax.add_patch(plt.Rectangle((z.xmin, z.ymin), z.xmax - z.xmin, z.ymax - z.ymin,
                                   facecolor=RACK, edgecolor=RACK_EDGE, lw=1.0,
                                   alpha=rack_alpha, zorder=2))
    if show_cameras:
        for c in lay.cameras:
            col = CAM_COLOUR[c.name]
            ax.plot(c.x, c.y, marker="o", ms=13, color=col, mec="white", mew=2.0, zorder=9)
            ang = np.radians(c.yaw_deg)
            ax.annotate("", xy=(c.x + 2.3 * np.cos(ang), c.y + 2.3 * np.sin(ang)),
                        xytext=(c.x, c.y), zorder=9,
                        arrowprops=dict(arrowstyle="-|>", lw=2.6, color=col,
                                        shrinkA=6, shrinkB=0))
            if camera_labels:
                ax.text(c.x + 2.9 * np.cos(ang), c.y + 2.9 * np.sin(ang), c.name,
                        color=col, fontsize=13, fontweight="bold",
                        ha="center", va="center", zorder=10)
    ax.set_xlim(-12.9, 12.9); ax.set_ylim(-10.9, 10.9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_support(ax, field, cell=0.67, vmin=0.0, vmax=1.0, hatch_zero=True):
    """Measured support, drawn as the cells it was actually measured in -- not interpolated."""
    for (x, y), v in field.items():
        ax.add_patch(plt.Rectangle((x - cell / 2, y - cell / 2), cell, cell,
                                   facecolor=SUPPORT((v - vmin) / max(vmax - vmin, 1e-9)),
                                   edgecolor="none", zorder=3))
        if hatch_zero and v <= 1e-9:
            ax.add_patch(plt.Rectangle((x - cell / 2, y - cell / 2), cell, cell,
                                       facecolor="none", edgecolor=BAD, lw=0.6,
                                       hatch="////", zorder=4, alpha=1.0))
    return plt.cm.ScalarMappable(cmap=SUPPORT,
                                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
