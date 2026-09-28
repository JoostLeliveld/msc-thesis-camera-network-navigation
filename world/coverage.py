#!/usr/bin/env python3
"""Coverage engine for the layout sketches.

Same physics as scripts/geometry_visibility: a 2.5-D height map, a pinhole
oblique camera, and a sampled sight-line from the camera to a marker riding at
0.35 m.  It is re-implemented here against a rasterised height map rather than
axis-aligned Prism objects, because two of the sketches rotate their racks and
`Prism` is axis-aligned by construction.  The camera itself is the project's own
`ObliqueCameraModel`, so the projection is identical to the runtime one.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/unav_common"))
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402

from layouts import HALL_X, HALL_Y, MARKER_Z, NOGO_MARGIN, Camera, Lane, Layout  # noqa: E402

IMG_W, IMG_H, FOV_H = 1280, 720, 1.5708
CELL = 0.10


def grid():
    xs = np.arange(HALL_X[0] + CELL / 2, HALL_X[1], CELL)
    ys = np.arange(HALL_Y[0] + CELL / 2, HALL_Y[1], CELL)
    return xs, ys


def _inside_rot_rect(gx, gy, cx, cy, sx, sy, yaw, pad=0.0):
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy = gx - cx, gy - cy
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    return (np.abs(lx) <= sx / 2 + pad) & (np.abs(ly) <= sy / 2 + pad)


def height_map(layout: Layout, state: str, xs, ys) -> np.ndarray:
    """What is PHYSICALLY there in this fill state. Not in the map."""
    gx, gy = np.meshgrid(xs, ys)
    H = np.zeros_like(gx)
    for o in (layout.fill_a if state == "A" else layout.fill_b):
        m = _inside_rot_rect(gx, gy, o.cx, o.cy, o.sx, o.sy, o.yaw)
        H = np.where(m, np.maximum(H, o.h), H)
    return H


def zone_mask(layout: Layout, xs, ys, pad=NOGO_MARGIN) -> np.ndarray:
    """The declared non-drivable zones. THIS is the map, and it is the same in
    every fill state."""
    gx, gy = np.meshgrid(xs, ys)
    M = np.zeros(gx.shape, bool)
    for z in layout.zones:
        M |= ((gx >= z.xmin - pad) & (gx <= z.xmax + pad) &
              (gy >= z.ymin - pad) & (gy <= z.ymax + pad))
    return M


SITE = (-11.35, 11.35, -9.35, 9.35)   # operating field; 0.65 m off every wall
# The chassis is 0.55 m wide, so a 0.55 m strip IS drivable; 0.70 m was discarding
# real floor. Lowered 2026-09-18 together with the target, because the greedy
# disjoint cover leaves pockets the first few big rectangles cannot reach - those
# pockets stranded 8 frozen task starts.
MIN_LANE_W = 0.55                      # thinner than the chassis is not drivable
LANE_TARGET = 0.995                    # stop once this much of reachable is covered
MIN_LANE_AREA_M2 = 0.30                # below this a rectangle is not worth a lane
MIN_PATCH_GAIN_M2 = 0.50               # a patch must declare this much NEW floor


def site_mask(xs, ys) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys)
    return (gx >= SITE[0]) & (gx <= SITE[1]) & (gy >= SITE[2]) & (gy <= SITE[3])


def free_mask(layout: Layout, xs, ys) -> np.ndarray:
    """Where the CHASSIS may be: hall interior minus real geometry, by half-width.

    Redrawn 2026-09-18. The previous definition was the site field minus the
    declared zones plus their 0.32 m planner envelope, which was wrong three ways:

    - it never looked at the collision geometry, so the five top-level <include>
      objects (bin_office, forklift_parked, pallet_jack and the two loose pallets)
      were invisible, and the derived lanes contained three of them;
    - SITE is hardcoded 0.65 m off every wall, stranding drivable floor;
    - the 0.32 m envelope is the PLANNER keep-out, not what the body needs.

    The margin is now the chassis half-width. The region says where the body may
    BE, not where it may spin: the planner checks its own footprint against it.
    """
    return _chassis_free_mask(xs, ys)


CHASSIS_HALF_WIDTH_M = 0.275
_COLLISION_INCLUDES = ("forklift_parked", "pallet_jack", "bin_office",
                       "pallet_loose_1", "pallet_loose_2")
_FREE_CACHE: dict = {}


def _chassis_free_mask(xs, ys) -> np.ndarray:
    """Cells whose centre clears every collision prism by the chassis half-width."""
    import sys as _sys
    from pathlib import Path as _Path
    key = (float(xs[0]), float(xs[-1]), len(xs), float(ys[0]), float(ys[-1]), len(ys))
    if key in _FREE_CACHE:
        return _FREE_CACHE[key]
    repo = _Path(__file__).resolve().parents[1]
    _sys.path.insert(0, str(repo / "src/unav_common"))
    from unav_common.occlusion_geometry import (
        parse_collision_scene_from_world, signed_distance_to_union_xy)
    world = repo / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    scene = parse_collision_scene_from_world(
        str(world), model_names=("warehouse_shell", "warehouse_v2_occluders"),
        include_names=_COLLISION_INCLUDES, robot_z_range=(0.0, 0.55))
    gx, gy = np.meshgrid(xs, ys)
    d = signed_distance_to_union_xy(
        scene.prisms, np.column_stack([gx.ravel(), gy.ravel()]), keep_in=False)
    M = (d >= CHASSIS_HALF_WIDTH_M).reshape(gx.shape)
    _FREE_CACHE[key] = M
    return M


def reachable_mask(layout: Layout, xs, ys) -> np.ndarray:
    """Largest 4-connected component of the free space.

    Pockets sealed behind a wall-backed rack are free but cannot be driven to,
    and counting them as covered floor would flatter every layout.
    """
    from scipy import ndimage
    F = free_mask(layout, xs, ys)
    lab, n = ndimage.label(F)
    if n == 0:
        return F
    sizes = ndimage.sum(F, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _largest_rect(M: np.ndarray):
    """Largest all-True axis-aligned rectangle (histogram method). Returns
    (r0, r1, c0, c1) inclusive, or None."""
    H, W = M.shape
    heights = np.zeros(W, int)
    best = (0, None)
    for r in range(H):
        heights = np.where(M[r], heights + 1, 0)
        stack = []
        for c in range(W + 1):
            h = heights[c] if c < W else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (c - s)
                if area > best[0]:
                    best = (area, (r - sh + 1, r, s, c - 1))
                start = s
            stack.append((start, h))
    return best[1]


def auto_lanes(layout: Layout, xs, ys, max_rects: int = 200) -> tuple[list[Lane], np.ndarray]:
    """Greedy maximal-rectangle cover of the reachable free space.

    The lanes are DERIVED from the rack geometry rather than typed by hand, so a
    lane can never cross a green no-go envelope: that is a property of the
    construction, not something to re-check. Rectangles may overlap each other;
    that is fine and matches how world_profiles.yaml already lists aisles.
    """
    R = reachable_mask(layout, xs, ys)
    remaining = R.copy()
    lanes, covered = [], np.zeros_like(R)
    minc = max(1, int(round(MIN_LANE_W / CELL)))
    for k in range(max_rects):
        box = _largest_rect(remaining)
        if box is None:
            break
        r0, r1, c0, c1 = box
        remaining[r0:r1 + 1, c0:c1 + 1] = False
        if (r1 - r0 + 1) < minc or (c1 - c0 + 1) < minc:
            continue          # too thin to drive: drop it, keep searching
        if (xs[c1] - xs[c0]) * (ys[r1] - ys[r0]) < MIN_LANE_AREA_M2:
            continue          # a sliver, not a lane
        # Edges at cell CENTRES: a cell is kept because its centre clears the
        # margin, so extending to the outer edge adds unverified space.
        lanes.append(Lane(f"lane_{k+1:02d}",
                          float(xs[c0]), float(xs[c1]),
                          float(ys[r0]), float(ys[r1])))
        covered[r0:r1 + 1, c0:c1 + 1] = True
        if covered.sum() >= LANE_TARGET * R.sum():
            break

    # Second pass: patch lanes for reachable floor the greedy cover stranded.
    #
    # Greedy maximal rectangles leave thin remnants along their own edges - the
    # leftovers are 0.2-0.5 m strips, all narrower than the chassis, so the first
    # pass discards them and real floor goes undeclared. That stranded 8 frozen
    # task starts. Here each uncovered pocket is re-grown into the FULL free mask
    # (not just the remnant), so a patch may overlap an existing lane; a patch is
    # kept only when it declares floor no lane already covers.
    lanes.extend(_patch_lanes(R, covered, xs, ys, len(lanes)))
    return lanes, R


def _patch_lanes(reachable, covered, xs, ys, start_index: int) -> list:
    """Grow drivable rectangles over reachable floor no lane covers yet."""
    minc = max(1, int(round(MIN_LANE_W / CELL)))
    out: list = []
    todo = reachable & ~covered
    guard = 0
    while todo.any() and guard < 200:
        guard += 1
        # Seed at the uncovered cell, then expand within the reachable mask.
        r, c = (int(i) for i in np.argwhere(todo)[0])
        r0 = r1 = r
        c0 = c1 = c
        grew = True
        while grew:
            grew = False
            for side in ("up", "down", "left", "right"):
                nr0, nr1, nc0, nc1 = r0, r1, c0, c1
                if side == "up" and r0 > 0:
                    nr0 -= 1
                elif side == "down" and r1 < reachable.shape[0] - 1:
                    nr1 += 1
                elif side == "left" and c0 > 0:
                    nc0 -= 1
                elif side == "right" and c1 < reachable.shape[1] - 1:
                    nc1 += 1
                else:
                    continue
                if reachable[nr0:nr1 + 1, nc0:nc1 + 1].all():
                    r0, r1, c0, c1 = nr0, nr1, nc0, nc1
                    grew = True
        todo[r0:r1 + 1, c0:c1 + 1] = False
        if (r1 - r0 + 1) < minc or (c1 - c0 + 1) < minc:
            continue
        if (xs[c1] - xs[c0]) * (ys[r1] - ys[r0]) < MIN_LANE_AREA_M2:
            continue
        # A patch must EARN its place: enough of it must be floor no lane declares
        # yet. Without this the grower expands freely into covered space and the
        # map becomes dozens of mutually overlapping rectangles.
        new_cells = int((reachable[r0:r1 + 1, c0:c1 + 1]
                         & ~covered[r0:r1 + 1, c0:c1 + 1]).sum())
        if new_cells * CELL * CELL < MIN_PATCH_GAIN_M2:
            continue
        covered[r0:r1 + 1, c0:c1 + 1] = True
        start_index += 1
        out.append(Lane(f"lane_{start_index:02d}",
                        float(xs[c0]), float(xs[c1]), float(ys[r0]), float(ys[r1])))
    return out


def lane_mask(layout: Layout, xs, ys) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys)
    M = np.zeros(gx.shape, bool)
    for L in layout.lanes:
        M |= (gx >= L.xmin) & (gx <= L.xmax) & (gy >= L.ymin) & (gy <= L.ymax)
    return M


def drivable_mask(layout: Layout, xs, ys) -> np.ndarray:
    """The declared lane union. Identical between stock states A and B by
    construction -- footprints do not move, only shelf heights do."""
    return lane_mask(layout, xs, ys)


def make_cam(c: Camera) -> ObliqueCameraModel:
    return ObliqueCameraModel(cam_pos=(c.x, c.y, c.z), look_at=c.look_at(),
                              img_width=IMG_W, img_height=IMG_H, fov_h_rad=FOV_H)


# Deliberately NOT a hard visibility gate. On this world the detector hits 98.2 %
# of poses with a sight-line across the whole grid at confidence 0.25, so any cut
# tight enough to matter would contradict a measurement already on record. It is
# a sanity floor only (10 px/m is a 0.35 m marker at ~3 px, past the hall's far
# corner); the useful resolution signal is reported as a statistic instead.
PX_PER_M_MIN = 10.0


def pixels_per_m(cam: ObliqueCameraModel, xs, ys) -> np.ndarray:
    """Worst-direction ground resolution, px per m, by central differences on the
    world->pixel map. Same quantity as geometry_visibility.projection_jacobian_scale."""
    gx, gy = np.meshgrid(xs, ys)
    z = np.full_like(gx, MARKER_Z)
    h = 0.02

    def uv(dx, dy):
        pts = np.stack([gx + dx, gy + dy, z], -1)
        d = (pts - np.asarray(cam.cam_pos, float)) @ cam.R.T
        zc = np.where(d[..., 2] > 1e-9, d[..., 2], 1.0)
        f = cam.K[0, 0]
        return f * d[..., 0] / zc + cam.K[0, 2], f * d[..., 1] / zc + cam.K[1, 2]

    ux1, vx1 = uv(h, 0); ux0, vx0 = uv(-h, 0)
    uy1, vy1 = uv(0, h); uy0, vy0 = uv(0, -h)
    J = np.stack([np.stack([(ux1 - ux0) / (2 * h), (uy1 - uy0) / (2 * h)], -1),
                  np.stack([(vx1 - vx0) / (2 * h), (vy1 - vy0) / (2 * h)], -1)], -2)
    return np.linalg.svd(J, compute_uv=False)[..., -1]


def visible_from(cam: ObliqueCameraModel, H: np.ndarray, xs, ys,
                 n_samples: int = 48) -> np.ndarray:
    """Boolean grid: marker at (x, y, MARKER_Z) is in image AND has a clear ray."""
    gx, gy = np.meshgrid(xs, ys)
    tgt = np.stack([gx, gy, np.full_like(gx, MARKER_Z)], -1).reshape(-1, 3)
    cp = np.asarray(cam.cam_pos, float)

    # in-image test
    d = (tgt - cp) @ cam.R.T
    z = d[:, 2]
    front = z > 1e-9
    f, cx, cy = cam.K[0, 0], cam.K[0, 2], cam.K[1, 2]
    zs = np.where(front, z, 1.0)
    u = f * d[:, 0] / zs + cx
    v = f * d[:, 1] / zs + cy
    in_img = front & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    in_img &= (pixels_per_m(cam, xs, ys) >= PX_PER_M_MIN).ravel()

    # sight-line test, in chunks so the sample tensor stays small
    ok = np.zeros(len(tgt), bool)
    t = np.linspace(0.03, 0.94, n_samples)[None, :, None]
    x0, y0 = xs[0], ys[0]
    idx = np.where(in_img)[0]
    for s in range(0, len(idx), 8000):
        sel = idx[s:s + 8000]
        P = cp[None, None, :] + t * (tgt[sel][:, None, :] - cp[None, None, :])
        ix = np.clip(((P[..., 0] - x0) / CELL).round().astype(int), 0, len(xs) - 1)
        iy = np.clip(((P[..., 1] - y0) / CELL).round().astype(int), 0, len(ys) - 1)
        ok[sel] = np.all(P[..., 2] > H[iy, ix] + 1e-6, axis=1)
    return (in_img & ok).reshape(gx.shape)


def coverage(layout: Layout, state: str, xs, ys):
    H = height_map(layout, state, xs, ys)
    cams = [make_cam(c) for c in layout.cameras]
    per_cam = [visible_from(c, H, xs, ys) for c in cams]
    n = np.sum(per_cam, axis=0) if per_cam else np.zeros(H.shape, int)
    # best ground resolution available at each cell from any camera that sees it
    ppm = np.zeros(H.shape)
    for c, m in zip(cams, per_cam):
        ppm = np.maximum(ppm, np.where(m, pixels_per_m(c, xs, ys), 0.0))
    return {"height": H, "per_cam": per_cam, "n_visible": n, "px_per_m": ppm}


def crossing_angles(layout: Layout, xs, ys, per_cam, drive):
    """Median pairwise bearing separation at cells seen by >= 2 cameras.

    From the two-camera result already on record: what pays is OPPOSITE, not
    perpendicular -- so a layout whose overlap sits near 180 deg is worth more
    than one with the same overlap area near 90 deg.
    """
    gx, gy = np.meshgrid(xs, ys)
    best = []
    cams = layout.cameras
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            m = per_cam[i] & per_cam[j] & drive
            if m.sum() == 0:
                continue
            a1 = np.arctan2(gy[m] - cams[i].y, gx[m] - cams[i].x)
            a2 = np.arctan2(gy[m] - cams[j].y, gx[m] - cams[j].x)
            d = np.abs(np.degrees(np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2))))
            best.append((f"{cams[i].name}-{cams[j].name}", int(m.sum()), float(np.median(d))))
    return sorted(best, key=lambda r: -r[1])


def analyse(layout: Layout, derive_lanes: bool = True):
    xs, ys = grid()
    reach = reachable_mask(layout, xs, ys)
    if derive_lanes:
        lanes, reach = auto_lanes(layout, xs, ys)
        layout.lanes = lanes                   # the derived map replaces any guess
    else:
        lanes = layout.lanes                   # use the world's own declared lanes
    drive = drivable_mask(layout, xs, ys)
    A = coverage(layout, "A", xs, ys)
    B = coverage(layout, "B", xs, ys)
    nd = int(drive.sum())
    cell_area = CELL * CELL

    def frac(n, k):
        return float((n[drive] >= k).mean()) if nd else 0.0

    flip = ((A["n_visible"] >= 1) != (B["n_visible"] >= 1)) & drive
    flip2 = ((A["n_visible"] >= 2) != (B["n_visible"] >= 2)) & drive
    dpairs = int(np.abs(A["n_visible"][drive].astype(int) - B["n_visible"][drive].astype(int)).sum())
    envelope_hit = int((drive & zone_mask(layout, xs, ys)).sum())

    return {
        "layout": layout,
        "xs": xs, "ys": ys, "drive": drive, "reach": reach, "A": A, "B": B,
        "n_drive": nd, "area": nd * cell_area,
        "reach_area": int(reach.sum()) * cell_area,
        "map_fidelity": float(nd / max(1, int(reach.sum()))),
        "n_lanes": len(lanes),
        "covA1": frac(A["n_visible"], 1), "covA2": frac(A["n_visible"], 2),
        "covB1": frac(B["n_visible"], 1), "covB2": frac(B["n_visible"], 2),
        "flip_frac": float(flip.sum() / nd) if nd else 0.0,
        "flip_cells": int(flip.sum()),
        "flip2_frac": float(flip2.sum() / nd) if nd else 0.0,
        "pair_delta": dpairs,
        "angles": crossing_angles(layout, xs, ys, A["per_cam"], drive),
        "px_per_m_median": float(np.median(A["px_per_m"][drive & (A["n_visible"] >= 1)]))
        if nd and (drive & (A["n_visible"] >= 1)).any() else 0.0,
        "envelope_hit": envelope_hit,
    }
