#!/usr/bin/env python3
"""Appendix figure: the driveable map the planner is actually given.

The map is not a picture of the warehouse. It is the union of the DECLARED traversable
lane rectangles in src/experiments/config/world_profiles.yaml, minus the collision boxes
the world file declares (every link named obs*), eroded by the planner's own clearance and
reduced to the largest connected component. What is inside a storage zone is deliberately
absent: the map declares the zone, not its contents, so it carries no visibility
information and does not change between stock states.

Everything drawn here is read from those two files at build time via the same
driveable() the route library uses, so the figure cannot drift from the planner.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / 'world'))
from paths import repo_root  # noqa: E402
import route_tasks as rt  # noqa: E402
# route_tasks has already put src/experiments on the path for this import
from experiments.core.world_profiles import load_world_profiles  # noqa: E402

REPO = repo_root()
OUT = REPO / 'logs/studies/thesis_setup_figure_20260908'

DRIVE = '#cfe3f5'      # the region the planner may occupy
DRIVE_EDGE = '#3f7fb5'
ZONE = '#f0e2c8'       # declared storage / structure, never driveable
ZONE_EDGE = '#b08a4a'
INK = '#1d2530'
CAM = '#1f6fb8'


def main() -> None:
    xs, ys, mask, n_obs = rt.driveable()
    profile = load_world_profiles(str(rt.PROFILES))['worlds'][rt.WORLD_KEY]
    regions = profile.get('known_2d_regions', []) or []

    boundary = next(r for r in regions
                    if str(r.get('type', '')).strip().lower() == 'site_boundary')
    zones = [r for r in regions
             if str(r.get('type', '')).strip().lower() == 'non_driveable_obstacle']
    lanes = [r for r in regions
             if str(r.get('type', '')).strip().lower() == 'traversable']

    area = float(mask.sum()) * rt.RES ** 2

    fig, ax = plt.subplots(figsize=(9.2, 7.9))

    # The mask as actual cells: contourf interpolates between cell centres and so
    # bleeds the region a half-cell into the zones it must never enter, which this
    # figure would then appear to contradict. pcolormesh draws the cells themselves.
    edge_x = np.concatenate([xs - rt.RES / 2, [xs[-1] + rt.RES / 2]])
    edge_y = np.concatenate([ys - rt.RES / 2, [ys[-1] + rt.RES / 2]])
    ax.pcolormesh(edge_x, edge_y, np.ma.masked_where(~mask, mask.astype(float)),
                  cmap=matplotlib.colors.ListedColormap([DRIVE]), vmin=0, vmax=1,
                  shading='flat', zorder=1)
    # one outline around the region, drawn on the same cell edges
    ax.contour(xs, ys, mask.astype(float), levels=[0.5],
               colors=[DRIVE_EDGE], linewidths=1.1, zorder=2, alpha=0.85)

    # the declared zones the map keeps the robot out of
    for zone in zones:
        ax.add_patch(Rectangle(
            (zone['xmin'], zone['ymin']),
            zone['xmax'] - zone['xmin'], zone['ymax'] - zone['ymin'],
            facecolor=ZONE, edgecolor=ZONE_EDGE, lw=1.0, zorder=3))
        # the zone names are the map's own ids; wrap the compound ones so a label
        # stays inside the box it names
        label = zone['name'].replace('_', '\n')
        width = zone['xmax'] - zone['xmin']
        # camera A stands on the dock office's south-west corner and its arrow
        # crosses the box, so that one label sits high in its box instead of centred
        label_y = (zone['ymax'] - 0.42 if zone['name'] == 'DOCK_OFFICE'
                   else 0.5 * (zone['ymin'] + zone['ymax']))
        ax.text(0.5 * (zone['xmin'] + zone['xmax']), label_y,
                label, fontsize=7.4 if width < 3.6 else 8.4, color='#6b5220',
                ha='center', va='center', zorder=4, linespacing=1.15)

    # the planner's outer bound
    ax.add_patch(Rectangle(
        (boundary['xmin'], boundary['ymin']),
        boundary['xmax'] - boundary['xmin'], boundary['ymax'] - boundary['ymin'],
        facecolor='none', edgecolor=INK, lw=1.4, ls=(0, (6, 3)), zorder=5))

    # the five cameras, from the world file itself
    for name, (x, y, yaw) in sorted(camera_poses().items()):
        ax.plot(x, y, marker='o', ms=9, color=CAM, mec='white', mew=1.6, zorder=7)
        ax.annotate('', xy=(x + 1.5 * np.cos(yaw), y + 1.5 * np.sin(yaw)),
                    xytext=(x, y), zorder=6,
                    arrowprops=dict(arrowstyle='-|>,head_width=0.28,head_length=0.6',
                                    lw=2.0, color=CAM, shrinkA=7, shrinkB=0))
        ax.text(x - 0.78 * np.cos(yaw), y - 0.78 * np.sin(yaw), name,
                fontsize=9.5, fontweight='bold', color='white', ha='center',
                va='center', zorder=8,
                bbox=dict(boxstyle='circle,pad=0.20', fc=CAM, ec='white', lw=1.3))

    ax.set(xlim=(-12.6, 12.6), ylim=(-10.6, 10.6), aspect='equal')
    ax.set_xlabel('east (m)', fontsize=10.5)
    ax.set_ylabel('north (m)', fontsize=10.5)
    ax.set_title(
        'The driveable map the planner is given: declared lanes only, '
        f'{area:.0f}' + r'$\,$m$^2$ once' '\n'
        f'shrunk by the robot\'s {rt.CLEARANCE_M:.2f} m clearance. '
        'Storage contents are deliberately absent.',
        fontsize=11.5, color=INK)
    ax.tick_params(labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    ax.legend(handles=[
        Patch(facecolor=DRIVE, edgecolor=DRIVE_EDGE,
              label=f'driveable: {len(lanes)} declared lanes, '
                    f'minus {n_obs} collision boxes, eroded {rt.CLEARANCE_M:.2f} m'),
        Patch(facecolor=ZONE, edgecolor=ZONE_EDGE,
              label='declared storage or structure: never driveable, contents not mapped'),
        Line2D([], [], color=INK, lw=1.4, ls=(0, (6, 3)),
               label='operating bound (not painted on the floor)'),
        Line2D([], [], color=CAM, marker='o', ms=8, lw=2.0,
               label='wall camera and the direction it looks'),
    ], loc='lower center', bbox_to_anchor=(0.5, -0.235), fontsize=9,
        frameon=False, ncol=1, handlelength=2.4)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'driveable_map.pdf', OUT / 'driveable_map.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {OUT / "driveable_map.pdf"}')
    print(f'  driveable area {area:.1f} m2 over {int(mask.sum())} cells at '
          f'{rt.RES:.2f} m, {len(lanes)} lanes, {len(zones)} zones, '
          f'{n_obs} collision boxes')


def camera_poses() -> dict[str, tuple[float, float, float]]:
    """Each camera's world x, y and yaw, read from the world file's include poses."""
    import re
    names = {'external_camera': 'A', 'external_camera_b': 'B', 'external_camera_c': 'C',
             'external_camera_d': 'D', 'external_camera_e': 'E'}
    poses = {}
    for _, model, pose in re.findall(
            r'<include><name>([^<]+)</name><uri>model://([^<]+)</uri><pose>([^<]+)</pose>',
            rt.WORLD_SDF.read_text()):
        if model in names:
            values = [float(v) for v in pose.split()]
            poses[names[model]] = (values[0], values[1], values[5])
    return poses


if __name__ == '__main__':
    main()
