#!/usr/bin/env python3
"""Route tasks for warehouse_v2, with a route-separation (G3) report per task.

WHY THIS EXISTS. A route comparison only carries information if the arms could
have disagreed. The retracted e4 campaign offered every arm two candidates 0.19 m
apart (15.05 m vs 15.17 m) and then reported that the arms agreed -- which was
guaranteed by construction. So the task set is generated together with the menu of
topologically distinct routes each task admits, and a task that cannot offer a real
choice is reported as failing the gate rather than quietly used.

THE MENU IS ARM-NEUTRAL. Candidates come from the lane geometry alone -- the
shortest path plus Yen-style deviations, each found by forbidding one small disc on
the base path -- and never from any availability or uncertainty field. Every arm is
later offered the same menu, so the menu cannot favour one.

    python3 world/route_tasks.py

Writes logs/studies/warehouse_v2/route_tasks/{route_tasks.json,g3_report.csv}.
Reads no detector outcome and starts no simulator.
"""

from __future__ import annotations

import csv
import heapq
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/experiments"))
sys.path.insert(0, str(REPO / "src/unav_common"))
from experiments.core.world_profiles import load_world_profiles  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warehouse_v2 import ARTERY, A_AISLE_X, C_AISLE_Y  # noqa: E402

WORLD_KEY = "warehouse_v2.world.sdf"
WORLD_SDF = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
PROFILES = REPO / "src/experiments/config/world_profiles.yaml"
OUT = REPO / "logs/studies/warehouse_v2/route_tasks"

RES = 0.10                  # grid resolution, m
# 0.35 m: the planner's actual `nogo_safe_distance`, which is 0.275 m of robot
# half-width plus a 0.075 m margin. It was 0.25 here while the world documentation
# described 0.32 and the planner used 0.35 -- three numbers for one physical quantity,
# and the route library was the one eroding least, so it offered corridors the planner
# would refuse to drive. Measured cost of standardising: connected driveable area
# 170.4 -> 145.1 m2, and the library holds at 92 tasks with 91 offering a real route
# choice. In-place rotation still needs 0.486 m (the circumscribed radius), which is a
# planner rule rather than a map erosion.
CLEARANCE_M = 0.35
MIN_SEPARATION_M = 8.0      # a task must be a real traverse, not a nudge
G3_MIN_ROUTES = 2           # two genuinely distinct corridors is a real choice
#: Two candidates count as different ROUTES only if they use different corridors.
#: Anchored on this layout's aisle pitch, not tuned: section A's picking aisles are
#: 3.70 m apart and section C's block aisles 3.65 m, so a route in a different
#: corridor differs by at least one pitch. Measured over 974 candidate pairs the
#: separation distribution has its gap at 2-3 m (9 pairs) with a mode at 3-4 m
#: (138), so 3.0 m sits in the gap. An earlier 0.50 m floor admitted 635 pairs
#: that were nudges inside ONE corridor -- the e4 failure mode (0.19 m) again.
G3_MIN_SEP_M = 3.00
MAX_ROUTES = 6
CUT_RADII_M = (0.75, 1.30, 2.20)   # a cut must close a ~1.3 m eroded aisle

#: Named places, chosen from the layout's own structure: the three dock doors,
#: both ends of both section-A picking aisles, both ends of both section-C block
#: aisles, the north cross aisle, and the artery. Snapped to the driveable mask.
#: Aisle waypoints are DERIVED from the layout's aisle centrelines rather than
#: typed in, so widening an aisle moves the route library with it instead of
#: leaving a waypoint parked inside the new racking.
_ARTERY_X = 0.5 * (ARTERY[0] + ARTERY[1])
WAYPOINTS = {
    "dock_w":      (-8.50, -8.80),   # west dock door mouth
    "dock_m":      (-1.50, -8.80),   # centre dock door, under camera B
    "dock_e":      (5.50, -8.80),    # east dock door
    "apron_w":     (-11.00, -6.60),  # apron, west end by the dock office
    "aisleA1_s":   (A_AISLE_X[0], -5.20),   # section A, aisle A1|A2, south mouth
    "aisleA1_n":   (A_AISLE_X[0], 8.10),    # ... north mouth, into the cross aisle
    "aisleA2_s":   (A_AISLE_X[1], -5.20),   # section A, aisle A2|A3, south mouth
    "aisleA2_n":   (A_AISLE_X[1], 8.10),
    "artery_s":    (_ARTERY_X, -6.00),      # main north-south artery, south end
    "artery_n":    (_ARTERY_X, 6.20),       # ... north end
    "aisleC_s_w":  (0.90, C_AISLE_Y[0]),    # section C, south block aisle, west end
    "aisleC_s_e":  (10.60, C_AISLE_Y[0]),   # ... east end
    "aisleC_n_w":  (0.90, C_AISLE_Y[1]),    # section C, north block aisle, west end
    "aisleC_n_e":  (10.60, C_AISLE_Y[1]),
    "xaisle_e":    (10.60, 6.48),    # east cross aisle, under camera D
    "bay_B_e":     (10.60, 8.60),    # north-east, east of section B racking
}


# --------------------------------------------------------------------------
# Driveable mask
# --------------------------------------------------------------------------

def driveable():
    """Declared traversable lanes minus collision obstacles, eroded by clearance."""
    prof = load_world_profiles(str(PROFILES))["worlds"][WORLD_KEY]
    xs = np.arange(-12.0, 12.0 + 1e-9, RES)
    ys = np.arange(-10.0, 10.0 + 1e-9, RES)
    gx, gy = np.meshgrid(xs, ys)

    mask = np.zeros_like(gx, dtype=bool)
    for r in prof.get("known_2d_regions", []) or []:
        if str(r.get("type", "")).strip().lower() != "traversable":
            continue
        mask |= ((gx >= r["xmin"]) & (gx <= r["xmax"])
                 & (gy >= r["ymin"]) & (gy <= r["ymax"]))

    sdf = WORLD_SDF.read_text()
    n_obs = 0
    for _lname, body in re.findall(r'<link name="(obs[a-zA-Z_0-9]*)">(.*?)</link>', sdf, re.S):
        p = re.search(r"<pose>([-0-9. ]+)</pose>", body)
        g = re.search(r"<box><size>([-0-9. ]+)</size></box>", body)
        if not (p and g):
            continue
        px, py = [float(v) for v in p.group(1).split()][:2]
        sx, sy = [float(v) for v in g.group(1).split()][:2]
        mask &= ~((gx >= px - sx / 2) & (gx <= px + sx / 2)
                  & (gy >= py - sy / 2) & (gy <= py + sy / 2))
        n_obs += 1

    k = int(round(CLEARANCE_M / RES))
    eroded = ndimage.binary_erosion(
        mask, ndimage.generate_binary_structure(2, 1), iterations=k)
    lab, ncomp = ndimage.label(eroded, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if ncomp > 1:                              # keep the largest component only
        sizes = ndimage.sum(eroded, lab, range(1, ncomp + 1))
        eroded = np.asarray(lab == (1 + int(np.argmax(sizes))), dtype=bool)
    return xs, ys, np.asarray(eroded, dtype=bool), n_obs


def snap(xs, ys, mask, xy):
    """Nearest driveable cell index to a metric point."""
    iy = np.clip(int(round((xy[1] - ys[0]) / RES)), 0, len(ys) - 1)
    ix = np.clip(int(round((xy[0] - xs[0]) / RES)), 0, len(xs) - 1)
    if mask[iy, ix]:
        return (iy, ix)
    free = np.argwhere(mask)
    d = (free[:, 0] - iy) ** 2 + (free[:, 1] - ix) ** 2
    return tuple(free[int(np.argmin(d))])


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

NEIGH = [(dy, dx, float(np.hypot(dy, dx)))
         for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def dijkstra(mask, start, goal):
    ny, nx = mask.shape
    dist = np.full(mask.shape, np.inf)
    prev = {}
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        if node == goal:
            break
        y, x = node
        for dy, dx, step in NEIGH:
            jy, jx = y + dy, x + dx
            if not (0 <= jy < ny and 0 <= jx < nx) or not mask[jy, jx]:
                continue
            nd = d + step * RES
            if nd < dist[jy, jx]:
                dist[jy, jx] = nd
                prev[(jy, jx)] = node
                heapq.heappush(pq, (nd, (jy, jx)))
    if not np.isfinite(dist[goal]):
        return None
    path, node = [goal], goal
    while node != start:
        node = prev[node]
        path.append(node)
    return path[::-1]


def to_xy(xs, ys, path):
    return np.array([[xs[c], ys[r]] for r, c in path], dtype=float)


def length_m(poly):
    return float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))


def hausdorff(a, b):
    """Symmetric Hausdorff distance between two polylines, in metres."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _disc(shape, centre, radius_m):
    """Boolean disc of the given metric radius about a grid cell."""
    k = int(np.ceil(radius_m / RES))
    yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
    stamp = (yy * yy + xx * xx) * RES * RES <= radius_m * radius_m
    out = np.zeros(shape, bool)
    y0, x0 = centre
    ys0, ys1 = max(0, y0 - k), min(shape[0], y0 + k + 1)
    xs0, xs1 = max(0, x0 - k), min(shape[1], x0 + k + 1)
    out[ys0:ys1, xs0:xs1] = stamp[(ys0 - y0 + k):(ys1 - y0 + k),
                                  (xs0 - x0 + k):(xs1 - x0 + k)]
    return out


def diverse_routes(mask, start, goal, xs, ys):
    """Shortest path, then Yen-style deviations.

    Blocking a corridor around the WHOLE base path closes every 1.5-1.8 m aisle
    in this warehouse at once and the re-solve fails, which is why the menu must
    be built from SINGLE cuts: forbid one small disc on the base path at a time,
    forcing a detour past that point while leaving the rest of the map open. With
    six independent cycles in the lane network that is enough to reach the genuinely
    distinct corridors.
    """
    base = dijkstra(mask, start, goal)
    if base is None:
        return []
    kept = [to_xy(xs, ys, base)]

    step = max(1, int(round(1.20 / RES)))          # a cut every ~1.2 m of path
    cut_cells = [c for c in base[step:-step:step]]
    for radius in CUT_RADII_M:
        for cell in cut_cells:
            if len(kept) >= MAX_ROUTES:
                return kept
            trial = mask & ~_disc(mask.shape, cell, radius)
            trial[start] = True
            trial[goal] = True
            path = dijkstra(trial, start, goal)
            if path is None:
                continue
            poly = to_xy(xs, ys, path)
            if min(hausdorff(poly, q) for q in kept) < G3_MIN_SEP_M:
                continue
            kept.append(poly)
    return kept


# --------------------------------------------------------------------------

def main():
    xs, ys, mask, n_obs = driveable()
    print(f"driveable after {CLEARANCE_M} m erosion: "
          f"{mask.sum() * RES * RES:.1f} m^2 ({n_obs} obstacles subtracted)")

    places = {}
    for name, xy in WAYPOINTS.items():
        iy, ix = snap(xs, ys, mask, xy)
        places[name] = {"idx": (int(iy), int(ix)), "xy": [float(xs[ix]), float(ys[iy])]}
    # drop waypoints that snapped onto the same cell as an earlier one
    seen, uniq = {}, {}
    for name, p in places.items():
        key = tuple(p["idx"])
        if key in seen:
            print(f"  dropped {name}: snapped onto {seen[key]}")
            continue
        seen[key] = name
        uniq[name] = p
    places = uniq
    print(f"{len(places)} distinct waypoints on the driveable mask")

    names = sorted(places)
    tasks, rows = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = np.array(places[a]["xy"]), np.array(places[b]["xy"])
            if float(np.linalg.norm(pa - pb)) < MIN_SEPARATION_M:
                continue
            routes = diverse_routes(mask, tuple(places[a]["idx"]),
                                    tuple(places[b]["idx"]), xs, ys)
            if not routes:
                print(f"  UNREACHABLE {a} -> {b}")
                continue
            lens = [length_m(r) for r in routes]
            seps = [hausdorff(routes[p], routes[q])
                    for p in range(len(routes)) for q in range(p + 1, len(routes))]
            max_sep = max(seps) if seps else 0.0
            passed = len(routes) >= G3_MIN_ROUTES and max_sep >= G3_MIN_SEP_M
            name = f"{a}__{b}"
            tasks.append({
                "name": name,
                "start": places[a]["xy"], "goal": places[b]["xy"],
                "n_routes": len(routes),
                "route_lengths_m": [round(v, 3) for v in lens],
                "shortest_m": round(min(lens), 3),
                "max_detour_frac": round(max(lens) / min(lens) - 1.0, 4),
                "max_separation_m": round(max_sep, 3),
                "g3_pass": bool(passed),
                "routes": [[[round(float(x), 3), round(float(y), 3)] for x, y in r]
                           for r in routes],
            })
            rows.append({k: tasks[-1][k] for k in
                         ("name", "n_routes", "shortest_m", "max_detour_frac",
                          "max_separation_m", "g3_pass")})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "route_tasks.json").write_text(json.dumps({
        "world": WORLD_KEY,
        "resolution_m": RES,
        "clearance_m": CLEARANCE_M,
        "min_separation_m": MIN_SEPARATION_M,
        "g3": {"min_routes": G3_MIN_ROUTES, "min_separation_m": G3_MIN_SEP_M},
        "menu_source": "diverse shortest paths on the lane geometry only; "
                       "no availability or uncertainty field is read",
        "waypoints": places,
        "tasks": tasks,
    }, indent=2))
    with open(OUT / "g3_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    npass = sum(r["g3_pass"] for r in rows)
    print(f"\n{len(rows)} tasks | G3 pass {npass} ({100 * npass / max(len(rows), 1):.0f}%)")
    print(f"median distinct routes {np.median([r['n_routes'] for r in rows]):.1f}, "
          f"median max separation {np.median([r['max_separation_m'] for r in rows]):.2f} m")
    print(f"wrote {OUT}/route_tasks.json and g3_report.csv")


if __name__ == "__main__":
    main()
