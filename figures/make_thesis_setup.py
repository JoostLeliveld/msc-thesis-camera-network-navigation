#!/usr/bin/env python3
"""The introduction's setup figure: the simulated warehouse seen from above.

A real Gazebo frame from the world's own plan-view camera, with the five wall cameras
marked and their viewing directions drawn.

The cameras themselves are invisible in the render: external_camera/model.sdf carries only
sensors, every one with <visualize>false</visualize>, so it has no visual geometry. The grey
discs in the frame are the sixteen ceiling lamps. Marker positions are therefore computed
from the plan camera's own intrinsics rather than measured off the image, and the projection
is checked by placing the four hall floor corners, which land on the rendered corners.

Plan camera, read from the running simulation with `ign model -m plan_view_camera --pose`
and `ign topic -e -t /plan_view_camera/camera_info`: 42 m above the origin looking straight
down, fx = fy = 1656.126 px, principal point (800, 600) in a 1600x1200 image.

Recapture the frame, if the world changes:

    ros2 launch sim bringup_sim.launch.py world:=warehouse_v2.world.sdf
    ign topic -e -t /plan_view_camera/image_raw -n 1 --json-output > plan.json

then decode it with scripts/paper_figures/capture_overview_frame.py.
"""
from __future__ import annotations

import pathlib
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import repo_root  # noqa: E402

REPO = repo_root()
WORLD = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'
PLAN = OUT / 'gazebo_plan_view.png'

CAM_MODELS = {
    'external_camera': 'A', 'external_camera_b': 'B', 'external_camera_c': 'C',
    'external_camera_d': 'D', 'external_camera_e': 'E',
}
PLAN_HEIGHT_M = 42.0
FX = FY = 1656.1258
PRINCIPAL_PX = (800.0, 600.0)
CROP = (185, 85, 1415, 1115)
ARROW_PX = 110
CAM = '#1f6fb8'


def camera_poses() -> dict[str, tuple[float, float, float]]:
    """Each camera's world x, y and yaw, from the world file's include poses."""
    poses = {}
    for _, model, pose in re.findall(
            r'<include><name>([^<]+)</name><uri>model://([^<]+)</uri><pose>([^<]+)</pose>',
            WORLD.read_text()):
        if model in CAM_MODELS:
            values = [float(v) for v in pose.split()]
            poses[CAM_MODELS[model]] = (values[0], values[1], values[5])
    return poses


def project(x: float, y: float, z: float) -> tuple[float, float]:
    """World point to pixel, for a camera looking straight down from PLAN_HEIGHT_M."""
    cx, cy = PRINCIPAL_PX
    depth = PLAN_HEIGHT_M - z
    return cx + FX * x / depth, cy - FY * y / depth


def main() -> None:
    if not PLAN.exists():
        raise SystemExit(f'missing {PLAN}; capture it from a running simulation first')

    left, top, right, bottom = CROP
    image = mpimg.imread(PLAN)[top:bottom, left:right]
    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(7.2, 7.2 * height / width))
    ax.imshow(image)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')

    for name, (x, y, yaw) in sorted(camera_poses().items()):
        ux, uy = project(x, y, 5.0)
        px, py = ux - left, uy - top
        # world +x runs right and +y runs up in this view, so image dy flips sign
        dx, dy = np.cos(yaw), -np.sin(yaw)
        ax.annotate('', xy=(px + ARROW_PX * dx, py + ARROW_PX * dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.34,head_length=0.7',
                                    lw=3.0, color=CAM, shrinkA=12, shrinkB=0))
        ax.plot(px, py, marker='o', ms=12, color=CAM, mec='white', mew=2.0, zorder=5)
        # label on the far side of the marker from its arrow, so the two never overlap
        ax.text(px - 40 * dx, py - 40 * dy, name, color='white', fontsize=12.5,
                fontweight='bold', ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='circle,pad=0.26', fc=CAM, ec='white', lw=1.8))

    fig.tight_layout(pad=0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'thesis_setup.pdf', OUT / 'thesis_setup.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    print('wrote', OUT / 'thesis_setup.pdf')


if __name__ == '__main__':
    main()
