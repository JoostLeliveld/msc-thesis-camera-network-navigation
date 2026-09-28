#!/usr/bin/env python3
"""Candidate warehouse-v2 layouts, as data.

THE MAP IS A ZONE MAP, NOT AN OBJECT MAP
----------------------------------------
The drivable map declares rectangular *storage zones* that the robot may not
enter. What sits inside a zone is a separate thing entirely: racks, pallet
stacks, a high-bay block, or nothing at all. Objects can be restacked, moved
within their zone, or cleared out, and the map does not change by one cell.

That is the property the study needs. Every layout ships two fill states, A and
B, over an identical zone map:

    drivable map   identical in A and B, to the cell
    what you see   completely different

So a planner that reads the map knows exactly where it may drive and learns
nothing at all about where it will be seen. Anything a model gets right about
visibility has to have come from observation, not from the floor plan.

Design rules every sketch obeys, from the supervisor's brief plus what is
already established here:

R1  Hall stays ~24 x 20 m. Walls go to 9.0 m -- WallB_01's native height and a
    realistic clear height. The current world has 4.5 m walls with cameras
    mounted at 6.1 m, i.e. floating 1.6 m above the wall they are bolted to.
R2  At most five cameras, deliberately not a mirrored set: different walls,
    irregular spacing, off-normal yaw, three different mount heights.
R3  Lanes are DERIVED from the zone map (site field, minus zones plus a 0.32 m
    envelope, largest reachable component, greedy rectangle cover), so a lane
    can never cross a no-go envelope by construction.
R4  Fill objects stay strictly inside their zone. Asserted in code.
R5  Stacks are high or low and they live inside zones, never loose in an aisle.
    Loose aisle obstacles were measured to be worth 1.4 % of newly-blind
    reachable cells here, and an obstacle big enough to darken an aisle severs it.

Geometry is in metres, world frame, hall centred on the origin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mesh_library import MESHES, colour, footprint, height

HALL_X = (-12.0, 12.0)
HALL_Y = (-10.0, 10.0)
WALL_H = 9.0
NOGO_MARGIN = 0.32
MARKER_Z = 0.35
RACK_W = 0.88            # ShelfD/E native depth

STOCK = {"empty": 0.0, "low": 1.10, "std": 2.09, "tall": 2.61, "high": 4.20, "bulk": 6.53}


# ---------------------------------------------------------------------------
@dataclass
class Zone:
    """A declared non-drivable storage zone. This, and only this, is in the map."""
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    kind: str = "storage"       # storage | structure | staging

    @property
    def cx(self): return 0.5 * (self.xmin + self.xmax)
    @property
    def cy(self): return 0.5 * (self.ymin + self.ymax)
    @property
    def sx(self): return self.xmax - self.xmin
    @property
    def sy(self): return self.ymax - self.ymin

    def contains(self, o: "Obstacle", tol=1e-6) -> bool:
        c = o.corners()
        return (c[:, 0].min() >= self.xmin - tol and c[:, 0].max() <= self.xmax + tol
                and c[:, 1].min() >= self.ymin - tol and c[:, 1].max() <= self.ymax + tol)


@dataclass
class Obstacle:
    """A physical thing inside a zone: footprint, top height, dressing mesh."""
    name: str
    cx: float
    cy: float
    sx: float
    sy: float
    yaw: float
    h: float
    mesh: str
    zone: str = ""

    def corners(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hx, hy = self.sx / 2.0, self.sy / 2.0
        loc = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        return loc @ np.array([[c, -s], [s, c]]).T + np.array([self.cx, self.cy])


@dataclass
class Camera:
    name: str
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    mount: str
    colour: str

    def look_at(self, d: float = 10.0):
        y, p = math.radians(self.yaw_deg), math.radians(self.pitch_deg)
        return (self.x + d * math.cos(p) * math.cos(y),
                self.y + d * math.cos(p) * math.sin(y),
                self.z - d * math.sin(p))


@dataclass
class Lane:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class Layout:
    key: str
    title: str
    idea: str
    tradeoff: str
    zones: list[Zone] = field(default_factory=list)
    fill_a: list[Obstacle] = field(default_factory=list)
    fill_b: list[Obstacle] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def check(self) -> list[str]:
        """Every fill object must lie inside its own zone. Returns violations."""
        zmap = {z.name: z for z in self.zones}
        bad = []
        for tag, fill in (("A", self.fill_a), ("B", self.fill_b)):
            for o in fill:
                z = zmap.get(o.zone)
                if z is None:
                    bad.append(f"{tag}:{o.name} has no zone {o.zone!r}")
                elif not z.contains(o):
                    bad.append(f"{tag}:{o.name} sticks out of zone {o.zone}")
        return bad


# ---------------------------------------------------------------------------
# Fill catalogue: what can be inside a storage zone
# ---------------------------------------------------------------------------
def _runs_along(z: Zone, n: int, h: float, mesh: str, span=(0.0, 1.0), inset=0.06):
    """`n` shelf runs laid along the zone's long axis, inside its short axis."""
    out = []
    long_x = z.sx >= z.sy
    Lspan, Wspan = (z.sx, z.sy) if long_x else (z.sy, z.sx)
    depth = min(RACK_W, (Wspan - 2 * inset) / n)
    t0 = (z.xmin if long_x else z.ymin) + span[0] * Lspan + inset
    t1 = (z.xmin if long_x else z.ymin) + span[1] * Lspan - inset
    if t1 - t0 < 0.4:
        return out
    for i in range(n):
        # centre the run pack inside the zone's short axis
        off = (Wspan - n * depth) / 2 + depth * (i + 0.5)
        w0 = (z.ymin if long_x else z.xmin) + off
        cl, sl = 0.5 * (t0 + t1), (t1 - t0)
        if long_x:
            out.append(Obstacle(f"{z.name}_r{i}", cl, w0, sl, depth, 0.0, h, mesh, z.name))
        else:
            out.append(Obstacle(f"{z.name}_r{i}", w0, cl, depth, sl, 0.0, h, mesh, z.name))
    return out


def _stacks(z: Zone, specs, inset=0.10):
    """Discrete pallet stacks inside a bay: (fraction along, mesh, n_stacked, yaw_deg).

    Each item is scaled so its ROTATED bounding box fits the bay's short axis,
    and its position is clamped so the rotated box stays inside the long axis.
    """
    out = []
    long_x = z.sx >= z.sy
    Lspan = (z.sx if long_x else z.sy) - 2 * inset
    Wspan = (z.sy if long_x else z.sx) - 2 * inset
    t_lo = (z.xmin if long_x else z.ymin) + inset
    for k, (t, mesh, n, yaw) in enumerate(specs):
        sx, sy = footprint(mesh)
        h = height(mesh) * n
        th = math.radians(yaw)
        ca, sa = abs(math.cos(th)), abs(math.sin(th))
        bx, by = sx * ca + sy * sa, sx * sa + sy * ca      # rotated bbox
        across = by if long_x else bx
        scale = min(1.0, Wspan / across) if across > 0 else 1.0
        sx, sy = sx * scale, sy * scale
        bx, by = bx * scale, by * scale
        along = bx if long_x else by
        if along > Lspan:
            continue
        pos = t_lo + min(max(t * Lspan, along / 2), Lspan - along / 2)
        if long_x:
            out.append(Obstacle(f"{z.name}_s{k}", pos, z.cy, sx, sy, th, h, mesh, z.name))
        else:
            out.append(Obstacle(f"{z.name}_s{k}", z.cx, pos, sx, sy, th, h, mesh, z.name))
    return out


def fill(z: Zone, variant: str) -> list[Obstacle]:
    """Build the physical contents of a zone. `variant` is the stock state."""
    if variant == "empty":                       # bay cleared out
        return []
    if variant == "picked":                      # bottom beam only
        return _runs_along(z, 1, STOCK["low"], "ShelfD_01")
    if variant == "std":
        return _runs_along(z, 1, STOCK["std"], "ShelfD_01")
    if variant == "tall":
        return _runs_along(z, 1, STOCK["tall"], "ShelfE_01")
    if variant == "high":
        return _runs_along(z, 1, STOCK["high"], "ShelfE_01")
    if variant == "bulk":                        # above every camera
        return _runs_along(z, 1, STOCK["bulk"], "ShelfF_01", inset=0.02)
    if variant == "half_tall":                   # loaded half, cleared half
        return _runs_along(z, 1, STOCK["tall"], "ShelfE_01", span=(0.0, 0.52))
    if variant == "half_tall_far":               # same, other end
        return _runs_along(z, 1, STOCK["tall"], "ShelfE_01", span=(0.48, 1.0))
    if variant == "double":                      # two shallow runs, gap between
        return _runs_along(z, 2, STOCK["std"], "ShelfD_01")
    if variant == "stacks_low":
        return _stacks(z, [(0.18, "ClutteringA_01", 1, 0), (0.52, "ClutteringA_01", 1, 12),
                           (0.84, "PalletJackB_01", 1, -8)])
    if variant == "stacks_high":
        return _stacks(z, [(0.20, "ClutteringC_01", 2, 5), (0.58, "ClutteringC_01", 2, -6),
                           (0.86, "ClutteringD_01", 1, 15)])
    if variant == "stacks_mixed":
        return _stacks(z, [(0.15, "ClutteringC_01", 2, 0), (0.50, "ClutteringA_01", 1, 20),
                           (0.82, "ClutteringC_01", 1, -12)])
    raise ValueError(f"unknown fill variant {variant!r}")


def structure(z: Zone, h: float, mesh="WallB_01") -> list[Obstacle]:
    return [Obstacle(f"{z.name}_body", z.cx, z.cy, z.sx - 0.04, z.sy - 0.04, 0.0, h, mesh, z.name)]


# ---------------------------------------------------------------------------
# zone-builder helpers
# ---------------------------------------------------------------------------
def col_bays(prefix, x_centre, width, y_breaks):
    """A north-south column of storage bays split by cross aisles."""
    return [Zone(f"{prefix}{i+1}", x_centre - width / 2, x_centre + width / 2, a, b)
            for i, (a, b) in enumerate(y_breaks)]


def row_bays(prefix, y_centre, depth, x_breaks):
    """An east-west row of storage bays."""
    return [Zone(f"{prefix}{i+1}", a, b, y_centre - depth / 2, y_centre + depth / 2)
            for i, (a, b) in enumerate(x_breaks)]


def assign(zones, variants_a, variants_b):
    """Attach fill state A and B to a list of zones (cycling the variant lists)."""
    A, B = [], []
    for i, z in enumerate(zones):
        A += fill(z, variants_a[i % len(variants_a)])
        B += fill(z, variants_b[i % len(variants_b)])
    return A, B


def angled(z: Zone, deg: float, h: float, mesh="ShelfE_01", depth=RACK_W, inset=0.10, n=1):
    """Runs sitting at an angle INSIDE an axis-aligned zone.

    The bay stays a rectangle in the map; only its contents are skewed. Longest
    run that still fits the zone's bounding box at this angle."""
    th = math.radians(deg)
    ca, sa = abs(math.cos(th)), abs(math.sin(th))
    W, H = z.sx - 2 * inset, z.sy - 2 * inset
    pack = n * depth                      # the runs are offset across, so the
    lim = []                              # bbox grows with the pack, not one run
    if ca > 1e-6:
        lim.append((W - pack * sa) / ca)
    if sa > 1e-6:
        lim.append((H - pack * ca) / sa)
    L = min(lim) if lim else 0.0
    if L < 0.5:
        return []
    out = []
    nx, ny = -math.sin(th), math.cos(th)     # across-run direction
    for i in range(n):
        off = depth * (i - (n - 1) / 2)
        out.append(Obstacle(f"{z.name}_a{i}", z.cx + nx * off, z.cy + ny * off,
                            L, depth, th, h, mesh, z.name))
    return out


# ---------------------------------------------------------------------------
# S1  Cross-grain
# ---------------------------------------------------------------------------
def sketch_crossgrain() -> Layout:
    L = Layout("S1", "Cross-grain",
        "The west half keeps north-south pick bays; the east half runs its bays "
        "east-west instead. A camera therefore looks straight down an aisle on "
        "one side of the hall and broadside at bay ends on the other.",
        "Cheapest break of symmetry, and every zone stays an axis-aligned "
        "rectangle, so world_profiles.yaml needs no new region type.")
    W = 1.40
    for k, (xc, breaks) in enumerate([
        (-11.30, [(-8.60, -3.40), (-1.60, 3.20), (4.40, 8.60)]),
        (-8.20,  [(-8.60, -4.60), (-2.80, 2.00), (3.20, 8.60)]),
        (-5.10,  [(-8.60, -3.40), (-1.60, 4.10), (5.30, 8.60)]),
        (-2.00,  [(-8.60, -5.40), (-3.60, 1.20), (2.40, 8.60)]),
    ]):
        L.zones += col_bays(f"W{k+1}", xc, W, breaks)
    for k, (yc, breaks) in enumerate([
        (-7.55, [(0.60, 5.60), (6.60, 11.30)]),
        (-4.45, [(0.60, 4.20), (5.20, 11.30)]),
        (-1.35, [(0.60, 6.40), (7.40, 11.30)]),
        (1.75,  [(0.60, 5.00), (6.00, 11.30)]),
        (4.85,  [(0.60, 6.90), (7.90, 11.30)]),
        (7.95,  [(0.60, 4.60), (5.60, 11.30)]),
    ]):
        L.zones += row_bays(f"E{k+1}", yc, W, breaks)
    L.zones.append(Zone("STG_s", -11.30, -7.60, -9.35, -8.85, kind="staging"))
    L.zones.append(Zone("STG_e", 11.00, 11.35, -6.00, 0.50, kind="staging"))

    va = ["std", "tall", "picked", "std", "half_tall", "stacks_low",
          "tall", "std", "picked", "double", "stacks_high", "std"]
    vb = ["tall", "picked", "bulk", "half_tall_far", "std", "stacks_high",
          "empty", "high", "std", "picked", "double", "tall"]
    L.fill_a, L.fill_b = assign(L.zones, va, vb)
    L.cameras = [
        Camera("A", -9.20, -9.85, 6.20, 78.0, 50.0, "south wall, west bay", "#2f80ed"),
        Camera("B", -1.60, 9.85, 5.00, -68.0, 42.0, "north wall, off-centre, low", "#27ae60"),
        Camera("C", 11.85, -4.20, 7.20, 152.0, 55.0, "east wall, high", "#9b51e0"),
        Camera("D", 4.80, 9.85, 6.20, -104.0, 47.0, "north wall, east bay", "#f2994a"),
        Camera("E", -11.85, 3.10, 4.60, -14.0, 36.0, "west wall, low, long throw", "#e05a5a"),
    ]
    L.notes = [
        "Bay pitch 3.10 m west (1.70 m aisles) against a 3.10 m row pitch east, "
        "but the cross-aisle breaks are at different places in every column.",
        "Fill state B empties one bay completely and turns another into 6.5 m "
        "bulk: the map cannot tell those two apart.",
    ]
    return L


# ---------------------------------------------------------------------------
# S2  Bulk spine
# ---------------------------------------------------------------------------
def sketch_bulkspine() -> Layout:
    L = Layout("S2", "Bulk spine",
        "One deep storage zone stands off-centre at x = +2.7 and splits the hall "
        "14.0 m / 7.7 m. In fill state A it holds a 6.5 m high-bay block, taller "
        "than every camera, so the split is total. In state B the same zone is "
        "half cleared and the hall reconnects visually without moving one cell "
        "of the map.",
        "Strongest occlusion structure the pack allows and the clearest handover "
        "story. Risk: the east strip may end up nearly uncovered in state A.")
    L.zones.append(Zone("SPINE", 1.55, 3.85, -8.60, 8.60))
    W = 1.40
    for k, (xc, breaks) in enumerate([
        (-11.30, [(-8.60, -3.90), (-2.10, 2.90), (4.10, 8.60)]),
        (-8.30,  [(-8.60, -4.90), (-3.10, 1.90), (3.10, 8.60)]),
        (-5.30,  [(-8.60, -3.30), (-1.50, 3.50), (4.70, 8.60)]),
        (-2.30,  [(-8.60, -5.60), (-3.80, 1.20), (2.40, 8.60)]),
        (0.40,   [(-8.60, -4.20), (-2.40, 2.60), (3.80, 8.60)]),
    ]):
        L.zones += col_bays(f"P{k+1}", xc, W, breaks)
    L.zones += col_bays("X", 11.30, W, [(-8.60, -3.60), (-1.80, 3.20), (4.40, 8.60)])
    L.zones.append(Zone("STG_s", -10.60, -6.40, -9.35, -8.85, kind="staging"))

    va = ["bulk"] + ["std", "picked", "tall", "double", "std", "half_tall",
                     "stacks_high", "tall", "picked", "std", "stacks_low", "tall"]
    vb = ["half_tall"] + ["tall", "std", "empty", "picked", "high", "std",
                          "stacks_low", "picked", "double", "tall", "stacks_high", "std"]
    L.fill_a, L.fill_b = assign(L.zones, va, vb)
    L.cameras = [
        Camera("A", -10.40, -9.85, 6.20, 72.0, 48.0, "south wall, far west", "#2f80ed"),
        Camera("B", -3.40, 9.85, 6.20, -84.0, 45.0, "north wall, west of spine", "#27ae60"),
        Camera("C", 11.85, 5.60, 5.20, -168.0, 40.0, "east wall, covers the strip", "#9b51e0"),
        Camera("D", 6.20, -9.85, 7.40, 104.0, 55.0, "south wall, high, east of spine", "#f2994a"),
        Camera("E", 11.85, -6.40, 5.20, 158.0, 40.0, "east wall, second strip camera", "#e05a5a"),
    ]
    L.notes = [
        "The spine's two ends are the only handover gates in fill state A.",
        "The A->B change here is one zone's contents, and it rewires the whole "
        "camera-handover graph.",
    ]
    return L


# ---------------------------------------------------------------------------
# S3  Notched hall
# ---------------------------------------------------------------------------
def sketch_notched() -> Layout:
    L = Layout("S3", "Notched hall",
        "A 7.4 x 6.0 m walled office with a mezzanine deck occupies the "
        "north-east corner, so the drivable floor is L-shaped rather than "
        "rectangular, and one camera is mounted on the office corner looking "
        "diagonally back across the hall.",
        "Most realistic of the five and the free space is genuinely non-convex. "
        "Costs a structure model the AWS pack does not contain, and the office "
        "is the one zone whose contents can never change.")
    office = Zone("OFFICE", 4.60, 12.00, 3.90, 9.90, kind="structure")
    L.zones.append(office)
    W = 1.40
    for k, (xc, breaks) in enumerate([
        (-11.30, [(-8.60, -3.10), (-1.30, 3.90), (5.10, 8.60)]),
        (-8.20,  [(-8.60, -4.30), (-2.50, 2.70), (3.90, 8.60)]),
        (-5.10,  [(-8.60, -3.10), (-1.30, 4.50), (5.70, 8.60)]),
    ]):
        L.zones += col_bays(f"N{k+1}", xc, W, breaks)
    for k, (yc, breaks) in enumerate([
        (-7.30, [(-2.60, 3.20), (4.40, 11.30)]),
        (-4.20, [(-2.60, 5.00), (6.20, 11.30)]),
        (-1.10, [(-2.60, 2.40), (3.60, 11.30)]),
        (2.00,  [(-2.60, 4.60), (5.80, 11.30)]),
    ]):
        L.zones += row_bays(f"C{k+1}", yc, W, breaks)
    L.zones.append(Zone("STG_s", -10.20, -5.80, -9.35, -8.85, kind="staging"))

    va = ["std", "picked", "tall", "double", "half_tall", "std",
          "stacks_low", "tall", "picked", "std", "stacks_high", "high"]
    vb = ["tall", "std", "empty", "picked", "std", "high",
          "stacks_high", "picked", "double", "tall", "stacks_low", "std"]
    L.fill_a, L.fill_b = assign([z for z in L.zones if z.kind != "structure"], va, vb)
    body = structure(office, 3.40)
    L.fill_a += body
    L.fill_b += body
    L.cameras = [
        Camera("A", -8.10, -9.85, 6.20, 66.0, 46.0, "south wall, west", "#2f80ed"),
        Camera("B", 2.90, -9.85, 6.20, 96.0, 44.0, "south wall, centre-east", "#27ae60"),
        Camera("C", -11.85, 5.40, 5.40, -22.0, 38.0, "west wall, oblique", "#9b51e0"),
        Camera("D", 4.85, 4.15, 4.10, 214.0, 34.0, "office south-west corner, low", "#f2994a"),
        Camera("E", -2.10, 9.85, 7.00, -76.0, 52.0, "north wall, high", "#e05a5a"),
    ]
    L.notes = [
        "The office is a permanent 3.4 m block: the one thing in the hall the "
        "map DOES predict, which is useful as a control.",
        "Camera D at 4.1 m on the office corner gives a viewing geometry no wall "
        "camera can reproduce.",
    ]
    return L


# ---------------------------------------------------------------------------
# S4  Fishbone
# ---------------------------------------------------------------------------
def sketch_fishbone() -> Layout:
    L = Layout("S4", "Fishbone",
        "Zones stay rectangular but their contents sit at an angle: the west "
        "block racks at +22 deg, the east block at -34 deg. Nothing in the hall "
        "is parallel to a wall or to anything else, and in fill state B the "
        "angles change as well as the heights.",
        "Widest spread of viewing angles, and it keeps the axis-aligned map "
        "because only the contents are skewed. Costs floor area: a rotated run "
        "needs a bigger rectangle around it.")
    # Four wide blocks with deliberately unequal gaps (1.30 / 1.90 / 1.70 / 1.60 m
    # of aisle), so no two aisles in the hall are the same width.
    for k, (xc, breaks) in enumerate([
        (-8.25, [(-8.60, -3.30), (-1.90, 3.40), (4.80, 8.60)]),
        (-2.75, [(-8.60, -4.60), (-3.20, 2.10), (3.50, 8.60)]),
    ]):
        L.zones += col_bays(f"F{k+1}", xc, 3.60, breaks)
    for k, (xc, breaks) in enumerate([
        (2.55, [(-8.60, -3.90), (-2.50, 2.80), (4.20, 8.60)]),
        (7.75, [(-8.60, -5.10), (-3.70, 1.60), (3.00, 8.60)]),
    ]):
        L.zones += col_bays(f"G{k+1}", xc, 3.60, breaks)
    L.zones += col_bays("G3", 10.65, 1.40, [(-8.60, -2.40), (-0.60, 8.60)])

    ang_a = {"F": 22.0, "G": -34.0}
    ang_b = {"F": 51.0, "G": -12.0}
    ha = ["std", "tall", "picked", "high", "std", "tall"]
    hb = ["tall", "picked", "std", "std", "high", "picked"]
    hmap = {"std": "std", "tall": "tall", "picked": "low", "high": "high"}
    for i, z in enumerate(L.zones):
        fam = z.name[0]
        n = 2 if min(z.sx, z.sy) > 2.4 else 1          # narrow bays take one run
        L.fill_a += angled(z, ang_a[fam], STOCK[hmap[ha[i % 6]]], n=n)
        if hb[i % 6] == "picked" and i % 5 == 0:
            continue                                   # this bay is cleared in B
        L.fill_b += angled(z, ang_b[fam], STOCK[hmap[hb[i % 6]]], n=n)
    L.cameras = [
        Camera("A", -6.40, -9.85, 6.20, 74.0, 47.0, "south wall", "#2f80ed"),
        Camera("B", 8.10, -9.85, 5.20, 118.0, 41.0, "south wall, east, low", "#27ae60"),
        Camera("C", -11.85, 1.20, 7.20, 6.0, 53.0, "west wall, high", "#9b51e0"),
        Camera("D", 1.30, 9.85, 6.20, -86.0, 45.0, "north wall", "#f2994a"),
    ]
    L.notes = [
        "Only four cameras, on three walls, none facing another.",
        "A->B rotates the racks inside the bays by 29 deg (west) and 22 deg "
        "(east): every sight-line changes, the map does not.",
    ]
    return L


# ---------------------------------------------------------------------------
# S5  Diagonal spine
# ---------------------------------------------------------------------------
def sketch_diagonal() -> Layout:
    L = Layout("S5", "Diagonal spine",
        "Traffic runs on a diagonal from the south-west dock to the north-east "
        "dispatch. Zones stay axis-aligned but step back row by row, so the main "
        "aisle is a staircase across the hall and the two leftover triangles hold "
        "blocks of different depth.",
        "Diagonal traffic and heavy asymmetry while every zone is still an "
        "axis-aligned rectangle. The staircase corners are the awkward part for "
        "the planner.")
    W = 1.40
    for i, xc in enumerate([-0.90, 1.90, 4.70, 7.50, 10.30]):
        top = 4.30 - 2.05 * i
        if top < -7.0:
            continue
        L.zones += col_bays(f"SE{i+1}", xc, W, [(-8.60, top - 3.30), (top - 2.20, top)])
    for i, yc in enumerate([2.30, 5.10, 7.90]):
        end = 2.00 - 2.90 * i
        L.zones += row_bays(f"NW{i+1}", yc, W, [(-11.30, end - 3.60), (end - 2.60, end)])
    L.zones += col_bays("WB", -11.30, W, [(-8.60, -4.20), (-3.10, 0.80)])
    L.zones.append(Zone("NEBULK", 6.60, 11.30, 6.90, 9.10))
    L.zones.append(Zone("STG_s", -9.80, -5.20, -9.35, -8.85, kind="staging"))

    va = ["tall", "std", "picked", "double", "high", "std",
          "half_tall", "tall", "stacks_low", "picked", "std", "stacks_high"]
    vb = ["picked", "high", "tall", "std", "std", "empty",
          "tall", "picked", "stacks_high", "double", "half_tall_far", "std"]
    L.fill_a, L.fill_b = assign(L.zones, va, vb)
    # the north-east bulk block is the constant: 6.5 m in both states
    L.fill_a = [o for o in L.fill_a if o.zone != "NEBULK"] + fill(
        [z for z in L.zones if z.name == "NEBULK"][0], "bulk")
    L.fill_b = [o for o in L.fill_b if o.zone != "NEBULK"] + fill(
        [z for z in L.zones if z.name == "NEBULK"][0], "bulk")
    L.cameras = [
        Camera("A", -10.20, -9.85, 6.20, 58.0, 44.0, "south-west dock corner", "#2f80ed"),
        Camera("B", -0.40, -9.85, 5.00, 108.0, 40.0, "south wall, low", "#27ae60"),
        Camera("C", 11.85, 1.40, 7.20, 172.0, 52.0, "east wall, high", "#9b51e0"),
        Camera("D", -11.85, -1.20, 6.20, 22.0, 45.0, "west wall", "#f2994a"),
        Camera("E", 6.90, 9.85, 5.60, -118.0, 42.0, "north wall, dispatch end", "#e05a5a"),
    ]
    L.notes = [
        "Cameras sit on all four walls and none faces another.",
        "The north-east bulk block is fixed at 6.5 m in both fill states, so it "
        "is the one occluder a map-only model could learn.",
    ]
    return L


ALL = [sketch_crossgrain, sketch_bulkspine, sketch_notched, sketch_fishbone, sketch_diagonal]


def load_all() -> list[Layout]:
    out = []
    for f in ALL:
        L = f()
        bad = L.check()
        if bad:
            raise AssertionError(f"{L.key}: fill escapes its zone: {bad[:4]}")
        out.append(L)
    return out
