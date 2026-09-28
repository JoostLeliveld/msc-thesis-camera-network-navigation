#!/usr/bin/env python3
"""Appendix figure: the planner map.

Drawn from the two files the planner reads. The site boundary and the keep-out regions
come from src/experiments/config/world_profiles.yaml: every collision footprint of
warehouse_v2, grown by 0.10 m on each side. The driveable region is the site boundary
minus the keep-out regions. The walls and obstacles are the collision boxes of the world
file, and the camera poses are its camera includes.
"""
from __future__ import annotations

import pathlib
import math
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
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
OUT = REPO / 'logs' / 'thesis' / 'figures'
DRIVE = '#dcebf8'
DRIVE_EDGE = '#3f7fb5'
KEEP = '#f3e6cf'
KEEP_EDGE = '#c98a1e'
SOLID = '#8a8f96'
INK = '#1d2530'
CAM = '#1f6fb8'
# keep-out boxes are grouped by the prefix of their collision name for the labels
GROUPS = {'A1': 'A1', 'A2': 'A2', 'A3': 'A3', 'B1': 'B1', 'Cn': 'Cn', 'Cm': 'Cm',
          'Cs': 'Cs', 'STAGE_MID': 'STAGE\nMID', 'STAGE_E': 'STAGE\nE',
          'dock_office': 'DOCK\nOFFICE'}


def group_of(name: str) -> str | None:
    short = re.sub(r'^.*?:(obs_)?', '', name)
    for key in sorted(GROUPS, key=len, reverse=True):
        if short.startswith(key):
            return key
    return None



def main() -> None:
    profile = load_world_profiles(str(rt.PROFILES))['worlds'][rt.WORLD_KEY]
    regions = profile['known_2d_regions']
    site = next(r for r in regions if r['type'] == 'site_boundary')
    keep = [r for r in regions if r['type'] == 'non_driveable_obstacle']

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    rect = lambda r, **kw: Rectangle((r['xmin'], r['ymin']), r['xmax'] - r['xmin'],
                                     r['ymax'] - r['ymin'], **kw)
    ax.add_patch(rect(site, facecolor=DRIVE, edgecolor='none', hatch='..',
                      lw=0, zorder=1))
    for r in keep:
        ax.add_patch(rect(r, facecolor=KEEP, edgecolor=KEEP_EDGE, lw=0.8, zorder=2))
    for x, y, sx, sy in solid_boxes():
        ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, facecolor=SOLID,
                               edgecolor='#50555c', lw=0.4, zorder=3))
    groups: dict[str, list[dict]] = {}
    for r in keep:
        key = group_of(r['name'])
        if key:
            groups.setdefault(key, []).append(r)
    for key, members in groups.items():
        x = 0.5 * (min(m['xmin'] for m in members) + max(m['xmax'] for m in members))
        y = 0.5 * (min(m['ymin'] for m in members) + max(m['ymax'] for m in members))
        ax.text(x, y, GROUPS[key], fontsize=7, ha='center', va='center', color=INK,
                zorder=5, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none',
                                    alpha=0.8))
    ax.add_patch(rect(site, facecolor='none', edgecolor=INK, lw=1.2, ls=(0, (6, 3)),
                      zorder=4))
    for name, (x, y, yaw) in sorted(camera_poses().items()):
        ax.annotate('', xy=(x + 1.4 * math.cos(yaw), y + 1.4 * math.sin(yaw)),
                    xytext=(x, y), zorder=6,
                    arrowprops=dict(arrowstyle='-|>,head_width=0.25,head_length=0.5',
                                    lw=1.8, color=CAM, shrinkA=5, shrinkB=0))
        ax.plot(x, y, 's', ms=6, color=CAM, mec='white', mew=1.0, zorder=7)
        ax.text(x - 0.7 * math.cos(yaw), y - 0.7 * math.sin(yaw), name, fontsize=8,
                fontweight='bold', color='white', ha='center', va='center', zorder=8,
                bbox=dict(boxstyle='circle,pad=0.18', fc=CAM, ec='white', lw=1.0))

    ax.set(xlim=(-12.4, 12.4), ylim=(-10.4, 10.4), aspect='equal')
    ax.set_xlabel('east (m)')
    ax.set_ylabel('north (m)')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=DRIVE, edgecolor=DRIVE_EDGE, hatch='..', label='driveable region'),
        Patch(facecolor=KEEP, edgecolor=KEEP_EDGE, label='keep-out region'),
        Patch(facecolor=SOLID, edgecolor='#50555c', label='walls and obstacles'),
        Line2D([], [], color=INK, lw=1.2, ls=(0, (6, 3)), label='site boundary'),
        Line2D([], [], color=CAM, marker='s', ms=6, lw=1.8,
               label='camera and viewing direction'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.09), fontsize=8, frameon=False,
        ncol=1)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'driveable_map.pdf', OUT / 'driveable_map.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {OUT / "driveable_map.pdf"}: {len(keep)} keep-out boxes, '
          f'{len(groups)} labelled groups')


def solid_boxes():
    """Axis-aligned collision boxes of the world's static links (walls and obstacles)."""
    out = []
    sdf = rt.WORLD_SDF.read_text()
    for body in re.findall(r'<link name="(?:obs[^"]*|wall[^"]*)">(.*?)</link>', sdf, re.S):
        p = re.search(r'<pose>([-0-9. e]+)</pose>', body)
        g = re.search(r'<collision.*?<box><size>([-0-9. e]+)</size></box>', body, re.S)
        if p and g:
            px, py = [float(v) for v in p.group(1).split()][:2]
            sx, sy = [float(v) for v in g.group(1).split()][:2]
            out.append((px, py, sx, sy))
    return out


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
