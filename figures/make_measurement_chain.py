#!/usr/bin/env python3
"""Problem-statement figure: how a camera image becomes a position measurement.

Three panels reading left to right, following one real detection from the frozen
capture:

  (a) the camera and the image it forms, with the detected box and the pixel
      the runtime selects from it,
  (b) the projection model, written out,
  (c) where that pixel lands on the driveable map the planner is given.

The layout follows the IWAI localization-pathway figure, which read better than the
stacked diagram it replaces: a camera icon with its viewing pyramid, the frame drawn
on a tilted image plane inside that pyramid, an explicit homography step, and a map
panel carrying the camera and its sightline to the back-projected point.

Everything is real. The frame, the box and the selected pixel are the capture's own.
The projection uses the same ObliqueCameraModel the runtime uses, built from the
camera's captured pose, and main() asserts that its floor intersection reproduces the
back-projection the dataset recorded. The map is the planner's driveable map, read
through the same route_tasks.driveable() as the appendix map figure and drawn in that
figure's colours, so the two agree by construction.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / 'src' / 'unav_common'))
from paths import repo_root  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

REPO = repo_root()
# The 09-09 master_capture predates the 09-18 world change (camera C tilt);
# recapture_v5 is the current-world capture the thesis is fitted on.
CAPTURE = REPO / 'logs/thesis/captures/v5/part1'
ADMITTED = REPO / 'logs/thesis/fits/gate_dataset/admitted.npz'
OUT = REPO.parent / 'papers' / 'Thesis' / 'figures'
POSE = 0              # an admitted current-capture pose for the drawn detection
CAMERA = 'camera_B'   # the capture's id; the paper calls it camera i
FOV_H_RAD = 1.5708    # external_camera/model.sdf

DETECT = '#c23d36'      # the detection and everything derived from it
INK = '#1d2530'
GREY = '#8a8a8a'
ARROW = '#2171b5'
# the appendix map figure's own palette, so the two maps read as the same map
DRIVE, DRIVE_EDGE = '#cfe3f5', '#3f7fb5'
ZONE, ZONE_EDGE, ZONE_TEXT = '#f0e2c8', '#b08a4a', '#6b5220'
CAM = '#1f6fb8'
HALO = [pe.withStroke(linewidth=3.2, foreground='white')]

# The image plane, drawn as a parallelogram in panel (a)'s own coordinates. The
# affine maps the frame's unit square onto it, exactly as the IWAI figure does.
IP_BL, IP_BR = np.array([2.30, 4.35]), np.array([6.95, 4.35])
IP_TL, IP_TR = np.array([3.55, 7.30]), np.array([8.20, 7.30])
AFF = Affine2D.from_values(*(IP_BR - IP_BL), *(IP_TL - IP_BL), *IP_BL)


def to_plane(points: np.ndarray) -> np.ndarray:
    """Unit-square frame coordinates onto the drawn image plane."""
    return AFF.transform(np.atleast_2d(points))


def camera_geometry() -> tuple[ObliqueCameraModel, np.ndarray]:
    """The real camera, built from its captured pose. Returns the model and its centre."""
    entry = next(c for c in json.loads((CAPTURE / 'capture_manifest.json').read_text())['cameras']
                 if c['camera_id'] == CAMERA)
    x, y, z, _, pitch, yaw = entry['pose_xyz_rpy']
    centre = np.array([x, y, z])
    axis = np.array([math.cos(pitch) * math.cos(yaw),
                     math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    look_at = centre + axis * (z / -axis[2])
    model = ObliqueCameraModel(cam_pos=centre, look_at=look_at,
                               img_width=entry['image_width'],
                               img_height=entry['image_height'], fov_h_rad=FOV_H_RAD)
    return model, centre


def driveable_map():
    """The planner's own driveable map, the same call the appendix map figure makes."""
    sys.path.insert(0, str(HERE.parents[1] / 'world'))
    import route_tasks as rt  # noqa: E402
    from experiments.core.world_profiles import load_world_profiles  # noqa: E402
    xs, ys, mask, _ = rt.driveable()
    regions = load_world_profiles(str(rt.PROFILES))['worlds'][rt.WORLD_KEY].get(
        'known_2d_regions', []) or []
    kind = lambda r: str(r.get('type', '')).strip().lower()  # noqa: E731
    zones = [r for r in regions if kind(r) == 'non_driveable_obstacle']
    boundary = next(r for r in regions if kind(r) == 'site_boundary')
    return xs, ys, mask, zones, boundary, rt.RES


def draw_camera_icon(ax, centre: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """A camera body at `centre`. Returns the lens position, where the rays start."""
    body = FancyBboxPatch((centre[0] - 0.62 * scale, centre[1] - 0.36 * scale),
                          1.10 * scale, 0.72 * scale, boxstyle='round,pad=0.06',
                          facecolor='#4a5058', edgecolor=INK, lw=1.1, zorder=6)
    ax.add_patch(body)
    ax.add_patch(FancyBboxPatch((centre[0] - 0.28 * scale, centre[1] + 0.30 * scale),
                                0.46 * scale, 0.22 * scale, boxstyle='round,pad=0.04',
                                facecolor='#4a5058', edgecolor=INK, lw=0.9, zorder=6))
    lens = centre + np.array([0.72 * scale, -0.06 * scale])
    ax.add_patch(Circle(tuple(lens), 0.30 * scale, facecolor='#4a5058',
                        edgecolor=INK, lw=1.2, zorder=7))
    ax.add_patch(Circle(tuple(lens), 0.17 * scale, facecolor='#9dc3e6',
                        edgecolor='none', zorder=8))
    ax.text(centre[0] - 0.15 * scale, centre[1] + 0.78 * scale, 'fixed camera $i$',
            ha='center', va='bottom', fontsize=10.5, color=INK, fontweight='bold')
    return lens


def panel_image(ax, row) -> None:
    """(a) the camera, the image it forms, and the pixel the runtime selects."""
    image = mpimg.imread(CAPTURE / row.image)
    height, width = image.shape[:2]
    ax.set(xlim=(0.15, 9.45), ylim=(2.55, 10.15))
    ax.axis('off')

    lens = draw_camera_icon(ax, np.array([1.60, 9.15]))
    for corner in (IP_BL, IP_BR, IP_TR, IP_TL):
        ax.plot([lens[0], corner[0]], [lens[1], corner[1]], '--', color=GREY,
                lw=1.0, zorder=1)

    # the real frame, mapped onto the tilted plane and clipped to it
    drawn = ax.imshow(np.flipud(image), extent=(0, 1, 0, 1), origin='lower',
                      aspect='auto', interpolation='bilinear',
                      transform=AFF + ax.transData, zorder=2)
    drawn.set_clip_path(Polygon([IP_BL, IP_BR, IP_TR, IP_TL], closed=True,
                                transform=ax.transData))
    ax.add_patch(Polygon([IP_BL, IP_BR, IP_TR, IP_TL], closed=True, fill=False,
                         edgecolor=CAM, lw=1.8, zorder=4))
    ax.text(IP_BR[0] + 0.10, IP_BR[1] - 0.12,
            r'camera image $I_{i,k}$', ha='right', va='top', fontsize=10,
            style='italic', color='#555b62')

    # the detection, in the frame's own normalised coordinates
    x0, y0, x1, y1 = (float(row[k]) for k in ('x0', 'y0', 'x1', 'y1'))
    box = to_plane(np.array([[x0 / width, 1 - y0 / height], [x1 / width, 1 - y0 / height],
                             [x1 / width, 1 - y1 / height], [x0 / width, 1 - y1 / height]]))
    ax.add_patch(Polygon(box, closed=True, fill=False, edgecolor=DETECT, lw=2.0,
                         zorder=6))
    u, v = (x0 + x1) / 2, y1
    pixel = to_plane(np.array([[u / width, 1 - v / height]]))[0]
    ax.plot(*pixel, '*', color=DETECT, ms=19, mec='white', mew=1.0, zorder=8)
    ax.annotate('bottom centre of the box',
                xy=pixel, xytext=(pixel[0] - 1.55, pixel[1] - 1.05), color=DETECT,
                fontsize=10, fontweight='bold', ha='center', va='top',
                path_effects=HALO, zorder=9,
                arrowprops=dict(arrowstyle='->', color=DETECT, lw=1.5))
    ax.text(pixel[0] - 1.55, pixel[1] - 1.46, r'$q^{\mathrm{box}}_{i,k}=(u,v)$',
            color=DETECT, fontsize=11.5, fontweight='bold', ha='center', va='top',
            path_effects=HALO, zorder=9)


def panel_model(ax) -> None:
    """(b) the projection model, written out."""
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')

    # The symbols are the paper's: the homography H_i of camera i, and the image
    # reference point and ground location the measurement chain names.
    ax.text(0.5, 0.70, 'inverse ground-plane\nhomography $H_i^{-1}$', ha='center',
            va='bottom', fontsize=12.5, color=INK, fontweight='bold')
    ax.add_patch(FancyArrowPatch((0.04, 0.62), (0.96, 0.62), arrowstyle='-|>',
                                 mutation_scale=26, lw=3.6, color=ARROW))
    ax.text(0.5, 0.45, r'$[\tilde x,\tilde y,\tilde w]^\top='
                       r'H_i^{-1}\,[u,v,1]^\top$',
            ha='center', va='center', fontsize=12.5, color=INK)
    ax.text(0.5, 0.30, r'$z_{i,k}=(\tilde x/\tilde w,\;\tilde y/\tilde w)$',
            ha='center', va='center', fontsize=12.5, color=INK)


def panel_map(ax, model, centre, row) -> np.ndarray:
    """(c) the driveable map, in the appendix figure's own style, with $z_k$ on it."""
    xs, ys, mask, zones, boundary, res = driveable_map()

    edge_x = np.concatenate([xs - res / 2, [xs[-1] + res / 2]])
    edge_y = np.concatenate([ys - res / 2, [ys[-1] + res / 2]])
    ax.pcolormesh(edge_x, edge_y, np.ma.masked_where(~mask, mask.astype(float)),
                  cmap=matplotlib.colors.ListedColormap([DRIVE]), vmin=0, vmax=1,
                  shading='flat', zorder=1, rasterized=True)
    ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=[DRIVE_EDGE],
               linewidths=1.0, zorder=2, alpha=0.85)
    for zone in zones:
        ax.add_patch(Rectangle((zone['xmin'], zone['ymin']),
                               zone['xmax'] - zone['xmin'], zone['ymax'] - zone['ymin'],
                               facecolor=ZONE, edgecolor=ZONE_EDGE, lw=0.9, zorder=3))
        # Only the rack runs are named. The dock-apron boxes are too small to hold a
        # label, and the appendix map figure carries the full set. Select on whether
        # the name fits the box, not on shape: the A runs are tall and narrow while
        # the B and C runs are wide and short, so any single dimension test drops some.
        span = min(zone['xmax'] - zone['xmin'], zone['ymax'] - zone['ymin'])
        if span > 1.2 and len(zone['name']) <= 3:
            ax.text(0.5 * (zone['xmin'] + zone['xmax']),
                    0.5 * (zone['ymin'] + zone['ymax']), zone['name'],
                    fontsize=7.6, color=ZONE_TEXT, ha='center', va='center', zorder=4)
    ax.add_patch(Rectangle((boundary['xmin'], boundary['ymin']),
                           boundary['xmax'] - boundary['xmin'],
                           boundary['ymax'] - boundary['ymin'], facecolor='none',
                           edgecolor=INK, lw=1.2, ls=(0, (6, 3)), zorder=4))

    ground = np.array(model.pixel_to_world((row.x0 + row.x1) / 2, row.y1))

    # the camera that took the frame, and its sightline to the point
    ax.plot([centre[0], ground[0]], [centre[1], ground[1]], ':', color='#6f767e',
            lw=1.5, zorder=5)
    ax.plot(centre[0], centre[1], 's', ms=8, color=CAM, mec='white', mew=1.3, zorder=7)
    look = model.look_at[:2] - centre[:2]
    look = look / np.linalg.norm(look)
    ax.annotate('', xy=centre[:2] + 2.1 * look, xytext=centre[:2], zorder=7,
                arrowprops=dict(arrowstyle='-|>,head_width=0.26,head_length=0.58',
                                lw=1.8, color=CAM, shrinkA=6, shrinkB=0))
    ax.text(centre[0] + 0.55, centre[1] + 0.55, 'camera $i$', fontsize=9.5, color=CAM,
            fontweight='bold', ha='left', va='bottom', path_effects=HALO, zorder=8)

    ax.plot(*ground, '*', color=DETECT, ms=22, mec='white', mew=1.1, zorder=9)
    ax.annotate(r'ground location $z_{i,k}$', xy=ground,
                xytext=(ground[0] - 1.2, ground[1] + 3.4), color=DETECT, fontsize=10,
                fontweight='bold', ha='center', va='bottom', path_effects=HALO,
                zorder=10, arrowprops=dict(arrowstyle='->', color=DETECT, lw=1.6))

    ax.set(xlim=(-12.9, 12.9), ylim=(-10.9, 10.9), aspect='equal')
    ax.set_xlabel('east (m)', fontsize=9.5)
    ax.set_ylabel('north (m)', fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return ground


def load_row():
    """The drawn detection: one admitted current-pipeline record and its frame."""
    admitted = np.load(ADMITTED, allow_pickle=False)
    selected = np.flatnonzero(
        (admitted['plan_pose_index'] == POSE) & (admitted['camera'] == CAMERA)
    )
    if len(selected) != 1:
        raise SystemExit(f'expected one admitted {CAMERA} observation at pose {POSE}, found {len(selected)}')
    row_index = int(selected[0])
    bbox = np.asarray(admitted['bbox_xyxy'][row_index], dtype=float)
    index = pd.read_csv(CAPTURE / 'capture_index.csv',
                        usecols=['image_sha1', 'image', 'camera_range_m'])
    image_sha1 = str(admitted['image_sha1'][row_index])
    match = index[index.image_sha1 == image_sha1]
    if len(match) != 1:
        raise SystemExit(f'expected one capture-index row for image {image_sha1}, found {len(match)}')
    record = match.iloc[0].copy()
    record['x0'], record['y0'], record['x1'], record['y1'] = bbox
    record['raw_x'], record['raw_y'] = np.asarray(admitted['raw_xy_m'][row_index], dtype=float)
    record['confidence'] = float(admitted['confidence'][row_index])
    return record


def main() -> None:
    row = load_row()
    model, centre = camera_geometry()

    # the drawing must reproduce the dataset's own back-projection, or the geometry lies
    recomputed = np.array(model.pixel_to_world((row.x0 + row.x1) / 2, row.y1))
    recorded = np.array([row.raw_x, row.raw_y])
    if not np.allclose(recomputed, recorded, atol=1e-4):
        raise SystemExit(f'camera model disagrees with the capture: {recomputed} vs {recorded}')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(13.0, 3.95), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(2.40, 1.18, 2.00))
    ax_image = fig.add_subplot(grid[0, 0])
    ax_model = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[0, 2])

    panel_image(ax_image, row)
    panel_model(ax_model)
    ground = panel_map(ax_map, model, centre, row)

    for ax, title in ((ax_image, '(a) YOLO image observation'),
                      (ax_model, '(b) projection model'),
                      (ax_map, '(c) position on the driveable map')):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)

    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'measurement_chain.pdf', OUT / 'measurement_chain.png'):
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT / 'measurement_chain.pdf')
    print(f'frame {row.image}, range {row.camera_range_m:.2f} m, confidence {row.confidence:.3f}')
    print(f'selected pixel ({(row.x0 + row.x1) / 2:.1f}, {row.y1:.1f}) -> '
          f'z_k ({ground[0]:.3f}, {ground[1]:.3f}) m')
    print(f'back-projection matches capture: {recomputed} == {recorded}')


if __name__ == '__main__':
    main()
