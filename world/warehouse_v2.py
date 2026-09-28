#!/usr/bin/env python3
"""ONE warehouse, three storage sections, nothing mirrored.

    SECTION A   west, selective pallet racking, runs north-south, 11.75 m long
    SECTION B   north-east, pallet racking turned 90 deg, shorter runs, lower
    SECTION C   south-east, block-stacked boxes on the floor -- no racking at
                all -- in three different stacking patterns

    plus an inbound dock apron along the south wall with a dock office in the
    corner, a main north-south artery between the west and east halves, and a
    cross aisle that only exists on the east side.

Why three sections. A real DC does not store everything the same way: fast full-
pallet movers go on the floor in block stacks, case-pick stock goes in selective
racking, and a second product family often sits in its own block turned the other
way. It is also the honest way to get asymmetry -- the building is different in
different places, rather than the same block scattered irregularly.

Nothing is mirrored: section widths differ, the two racking sections have
different run lengths (3 modules west, 2 modules north-east) and different top
beams, the dock doors are unevenly spaced, and the north cross aisle is 2.90 m in
the west and does not exist in the east.

Realism decisions, each with a reason:

- Racking is back-to-back PAIRS: 0.88 + 0.14 flue + 0.88 = 1.90 m, as built.
- Runs are whole ShelfD/E modules at their native 3.917 m, tiled, never stretched,
  so the camera sees the shelf the detector was trained on.
- Block stacks are whole Cluttering pallets at native size, stacked 1, 2 or 3 high.
- Walls are 9.0 m (WallB_01's native height). Cameras hang at 3.6-6.0 m, below the
  wall top. The current world mounts them at 6.10 m on 4.50 m walls.

The map declares the section zones. It says nothing about what is in them, and
the two stock states below prove it: same map to the cell, different building to
every camera.
"""
from __future__ import annotations

import math

from layouts import Camera, Layout, Obstacle, Zone
from mesh_library import footprint, height

# --------------------------------------------------------------------------- #
MODULE = 3.917           # ShelfD/E native length
DEPTH = 0.88             # ShelfD/E native depth
FLUE = 0.14
PAIR = 2 * DEPTH + FLUE  # 1.90 m rack zone

SITE = (-11.35, 11.35, -9.35, 9.35)
APRON_TOP = -4.40        # dock apron is everything south of this (4.95 m deep)
DOCK_DOORS = [-8.5, -1.5, 5.5]      # unevenly spaced on purpose

# --- section A: west, north-south racking, three back-to-back pairs -----------
# Aisle widths are the one dimension the ROBOT sets, so they are parameters
# rather than typed-in coordinates. The AMR is a real low-deck class: 0.80 x
# 0.55 m, which is a MiR250 (0.80 x 0.58) or an OTTO 100 (0.74 x 0.55) and needs
# a 0.971 m circle to turn on the spot. The declared keep-out envelope eats
# 0.32 m of each side of every aisle, so a nominal width w leaves the planner
# w - 0.64 m.
#
# The aisles were widened to fit that robot rather than the robot shrunk to fit
# the aisles, and the reason is measurable: at 0.55 m wide the robot spans 28.7
# px across the median covered cell, at 0.45 m only 23.5 px, and the whole
# measurement chain in this thesis starts with a detector finding it. Making the
# robot small is buying clearance with the evidence.
A_AISLE = 2.00                      # 1.36 m declared-clear: turn plus 0.39 m
A_X = [-8.90 + i * (PAIR + A_AISLE) for i in range(3)]
#: centreline of each section-A picking aisle, west to east
A_AISLE_X = [0.5 * (A_X[i] + A_X[i + 1]) for i in range(len(A_X) - 1)]
A_Y0, A_Y1 = APRON_TOP, APRON_TOP + 3 * MODULE          # -4.40 .. 7.35
# Top beam is TWO whole ShelfD modules, 5.23 m: 58 % of the 9.0 m clear height
# instead of 47 %, and no squashed part-tier in the render. Going taller still
# was measured and costs only 0.7 points of two-camera cover, so the limit here
# is what keeps the camera views usable, not the coverage.
A_TOP = {"full": 2 * 2.613, "half": 2.613, "picked": 1.10}

# --- main artery and the east half -------------------------------------------
#: main artery: from the east face of section A to the west edge of section C.
#: Widening the picking aisles takes its width from 2.35 m to 1.95 m, which still
#: leaves 1.31 m declared-clear against the AMR's 0.832 m turning circle.
ARTERY = (-8.90 + 2 * (PAIR + A_AISLE) + PAIR / 2, 1.80)
B_X0, B_X1 = 1.80, 1.80 + 2 * MODULE                    # 1.80 .. 9.63
# Section B gives up its second rack pair so section C can have three rows and
# two full-length aisles. One pair still reads as "racking turned 90 degrees",
# and the floor it releases is what the east cameras were short of.
B_Y = [8.35]
B_Y0, B_Y1 = 7.40, 9.30
B_TOP = {"full": 2.613, "half": 1.90, "picked": 1.10}   # one module, lower

CROSS_E = (5.56, 7.40)   # east cross aisle; its south edge is where section C ends

# --- section C: block-stacked boxes, three patterns --------------------------
# Floor storage as two LONG east-west rows with one long aisle between them,
# rather than a grid of little bays. The aisle then runs the full 9.55 m width of
# the section instead of stopping every 2.35 m, which is how block storage is
# actually laid out and gives the robot a proper run. Stack height varies ALONG
# each row, so one, two and three high are all visible from one aisle.
# The rows stop 1.50 m short of the site edge so there is an east perimeter lane.
# Without it the stacks ran into the wall camera's face and it saw 14 % of the floor.
C_X0, C_X1 = 1.80, 9.85                  # 8.05 m of row, east lane 9.85 .. 11.35
# Row depth is set by the pallet, not by taste: ClutteringA_01 is 1.983 m across
# its short axis and _block_row insets 0.06 m a side, so 2.12 m is the shallowest
# row the pallet actually fits in. The aisles then get everything that is left.
C_ROW_DEPTH = 2.12
C_AISLE = 1.80   # was 1.50, in which the AMR could not turn: it needs 0.971 m
                 # and a 1.50 m aisle offers 0.80 m of inscribed clearance
_c_names = ["Cs", "Cm", "Cn"]
C_ROWS, C_AISLES, _y = [], [], APRON_TOP
for _i, _nm in enumerate(_c_names):
    C_ROWS.append((_nm, _y, _y + C_ROW_DEPTH))
    _y += C_ROW_DEPTH
    if _i < len(_c_names) - 1:
        C_AISLES.append((_y, _y + C_AISLE))
        _y += C_AISLE
#: centreline of each section-C block aisle, south to north
C_AISLE_Y = [0.5 * (a + b) for a, b in C_AISLES]
# Three rows, three different product families -- three different meshes. The
# AWS pack has three cluttering loads and using one of them nine times is what
# made the block area read as copy-paste. They are different sizes as well as
# different shapes, so each row also stacks to its own height.
# Three rows, three genuinely different goods. Cardboard everywhere was the
# complaint and it was fair: a real DC holds cartons, steel drums and returnable
# plastic totes side by side, and they look nothing like each other.
#
#   drums  200 L steel drum, 0.58 m dia x 0.88 m -- 2x2 on a pallet = 1.20 m
#          square. Built from primitives on the metal PBR set, in pool colours.
#   totes  Large_Crate (OpenRobotics, CC-BY 4.0), 1.08 x 0.55 x 0.49 m, black
#          plastic. 2 x 3 to a pallet layer = 2.16 x 1.65 m.
DRUM_D, DRUM_H = 0.58, 0.88
TOTE_L, TOTE_W, TOTE_H = 1.08, 0.55, 0.49
C_MESH = {"Cs": "ClutteringA_01",                          # cartons, 1.06 m units
          "Cm": ("drums", 1.20, 1.20, DRUM_H),             # steel drums
          "Cn": ("totes", 2.16, 1.65, TOTE_H)}             # plastic totes
C_PALLET = "ClutteringA_01"           # kept: the apron staging still uses it

# Peak vs post-peak. Rows are the racking runs, columns the bays along each run.
A_PEAK = [["full", "full", "full"],
          ["full", "half", "full"],
          ["full", "full", "half"]]
A_AFTER = [["picked", "picked", "half"],
           ["picked", "picked", "picked"],
           ["half", "empty", "picked"]]      # middle bay out for a beam repair
B_PEAK = [["full", "full"]]
B_AFTER = [["picked", "half"]]

# Four stacks along each row, and how many pallets high each one is. Peak is a
# full floor; after the ship-out two positions are cleared and the rest come down.
# Stack counts are per row now, because the meshes are different heights: three
# ClutteringC loads would stand 5.16 m, taller than the racking and taller than
# anything block-stacked in a real DC. Counts are chosen so every row still has
# a tall, a medium and a short position but nothing exceeds 3.44 m.
# Counts are per row because the unit heights differ by 2x: a carton pallet is
# 1.06 m, a drum 0.88 m, a tote 0.49 m. Peak tops out at 3.17 m (cartons),
# 2.64 m (drums, 3 high is the real limit for stacked drums) and 1.96 m (totes).
C_PEAK = {"Cs": [1, 2, 3], "Cm": [3, 1, 2], "Cn": [4, 2, 3]}
C_AFTER = {"Cs": [0, 1, 2], "Cm": [1, 0, 2], "Cn": [2, 3, 0]}


# --------------------------------------------------------------------------- #
def _rack_pair(name, x, y0, y1, bays, tops, mesh_off=0):
    """Two faces of a back-to-back pair, one entry per bay along the run."""
    out = []
    for b, level in enumerate(bays):
        if level == "empty":
            continue
        yc = y0 + MODULE * (b + 0.5)
        for face, dx in (("w", -(FLUE / 2 + DEPTH / 2)), ("e", +(FLUE / 2 + DEPTH / 2))):
            mesh = "ShelfD_01" if (b + mesh_off) % 2 == 0 else "ShelfE_01"
            out.append(Obstacle(f"{name}b{b+1}{face}", x + dx, yc, DEPTH, MODULE,
                                0.0, tops[level], mesh, name))
    return out


def _rack_pair_ew(name, y, x0, x1, bays, tops, mesh_off=0):
    """Same, for a run turned 90 degrees."""
    out = []
    for b, level in enumerate(bays):
        if level == "empty":
            continue
        xc = x0 + MODULE * (b + 0.5)
        for face, dy in (("s", -(FLUE / 2 + DEPTH / 2)), ("n", +(FLUE / 2 + DEPTH / 2))):
            mesh = "ShelfD_01" if (b + mesh_off) % 2 == 0 else "ShelfE_01"
            out.append(Obstacle(f"{name}b{b+1}{face}", xc, y + dy, MODULE, DEPTH,
                                0.0, tops[level], mesh, name))
    return out


def _block_row(z: Zone, spec, highs: list[int]):
    """Whole unit loads set out along a storage row, each stacked its own number
    high. Positions are evenly spaced across the row so the gaps read as the
    working clearance between stacks.

    `spec` is either an AWS mesh name, or a (kind, sx, sy, unit_h) tuple for a
    load built from primitives and included props. The declared height always
    comes from the unit's REAL height times the stack count, so the occluder the
    sight-line model sees is the size of the thing that is drawn -- that is the
    whole point of deriving it rather than typing it in.
    """
    if isinstance(spec, tuple):
        kind, sx, sy, unit_h = spec
        mesh = kind
    else:
        mesh = spec
        sx, sy = footprint(spec)
        unit_h = height(spec)
    inset = 0.06
    if sy > z.sy - 2 * inset:               # turn the pallet to fit the row depth
        sx, sy = sy, sx
    if sy > z.sy - 2 * inset:
        raise AssertionError(f"pallet {sy:.2f} m deep does not fit row {z.name} "
                             f"({z.sy:.2f} m)")
    n = len(highs)
    span = z.sx - 2 * inset
    if n * sx > span:
        raise AssertionError(f"{n} pallets ({n*sx:.2f} m) do not fit row {z.name} "
                             f"({span:.2f} m)")
    pitch = span / n
    out = []
    for k, nh in enumerate(highs):
        if nh < 1:
            continue                        # this position is empty
        cx = z.xmin + inset + pitch * (k + 0.5)
        out.append(Obstacle(f"{z.name}_p{k}", cx, z.cy, sx, sy, 0.0,
                            unit_h * nh, mesh, z.name))
    return out


def build() -> Layout:
    L = Layout(
        "V2", "warehouse_v2",
        "Three storage sections that work differently: north-south selective "
        "racking in the west, a lower rack block turned 90 degrees in the "
        "north-east, and floor block-stacking in the south-east where the boxes "
        "are stacked one, two and three high in three separate patterns.",
        "Asymmetry comes from the building being genuinely different in "
        "different places, which is also what makes the stock change bite: the "
        "block-stack section can lose two metres of height without the map "
        "moving a cell.")

    # ---- zones: this, and only this, is what the map declares ----------------
    for i, x in enumerate(A_X):
        L.zones.append(Zone(f"A{i+1}", x - PAIR / 2, x + PAIR / 2, A_Y0, A_Y1))
    for i, y in enumerate(B_Y):
        L.zones.append(Zone(f"B{i+1}", B_X0, B_X1, y - PAIR / 2, y + PAIR / 2))
    for name, y0, y1 in C_ROWS:
        L.zones.append(Zone(name, C_X0, C_X1, y0, y1, kind="block_stack"))
    L.zones += [
        Zone("DOCK_OFFICE", -11.35, -8.85, -9.35, -7.55, kind="structure"),
        Zone("STAGE_MID", -4.60, -2.20, -9.35, -8.15, kind="staging"),
        Zone("STAGE_E", 2.60, 5.00, -9.35, -8.15, kind="staging"),
    ]

    # ---- contents: not declared anywhere ------------------------------------
    for tag, a_state, b_state, c_state in (("fill_a", A_PEAK, B_PEAK, C_PEAK),
                                           ("fill_b", A_AFTER, B_AFTER, C_AFTER)):
        objs = []
        for i, x in enumerate(A_X):
            objs += _rack_pair(f"A{i+1}", x, A_Y0, A_Y1, a_state[i], A_TOP, i)
        for i, y in enumerate(B_Y):
            objs += _rack_pair_ew(f"B{i+1}", y, B_X0, B_X1, b_state[i], B_TOP, i + 1)
        for name, _y0, _y1 in C_ROWS:
            z = [q for q in L.zones if q.name == name][0]
            objs += _block_row(z, C_MESH[name], c_state[name])
        # dock office and staged pallets on the apron
        oz = [q for q in L.zones if q.name == "DOCK_OFFICE"][0]
        objs.append(Obstacle("dock_office", oz.cx, oz.cy, oz.sx - 0.06, oz.sy - 0.06,
                             0.0, 2.80, "WallB_01", "DOCK_OFFICE"))
        peak = tag == "fill_a"
        for nm, h in (("STAGE_MID", 1.72 if peak else 1.06),
                      ("STAGE_E", 1.06 if peak else 1.72)):
            z = [q for q in L.zones if q.name == nm][0]
            for k in range(2):
                objs.append(Obstacle(f"{nm}{k}", z.xmin + 0.70 + 1.05 * k, z.cy,
                                     0.95, 1.00, 0.0, h, "ClutteringC_01", nm))
        setattr(L, tag, objs)

    # ---- cameras: four mounts, four heights, no mirrored pair ---------------
    # Chosen by measurement, not taste: seven placements were ray-cast against
    # this geometry. A low 3.6 m mount on the dock-office roof looked plausible
    # and cost 12 points of two-camera cover -- too grazing. Four corners plus a
    # dock camera scored the same but is the mirrored set this layout exists to
    # avoid. This set has no north-east camera, so it is not a mirrored pair.
    # MOUNT HEIGHT IS CAPPED AT 5.0 m -- a constraint, not an optimum. Measured
    # against this geometry it costs about 8 points of single-camera and 18 points
    # of two-camera cover against an 8.0 m mount. Three consequences worth knowing
    # rather than rediscovering:
    #
    #  * At 5 m the racking is TALLER than the cameras, and rack height above
    #    ~2.6 m then makes no difference at all to what floor is visible: a 2.61 m
    #    rack blocks an oblique view into a 1.80 m aisle exactly as completely as
    #    a 5.23 m one. The stock lever therefore comes from the floor storage and
    #    from bays picked down BELOW the camera line, not from the difference
    #    between a full and a half-full rack.
    #  * Shallower pitch pays down here: -4 degrees on every camera is worth 11
    #    points of two-camera cover, because reach matters more than look-down
    #    angle once the camera is low.
    #  * Aisle-end mounts are much worse at this height (2 % two-camera cover): a
    #    5 m camera looking along a 1.80 m aisle sees rack faces, not floor. The
    #    corner-and-dock arrangement survives the height change; camera C is
    #    turned to -50 deg, which is worth 3 points.
    L.cameras = [
        Camera("A", -11.45, -9.45, 5.00, 45.0, 44.0,
               "south-west corner, apron and the foot of section A", "#2a78d6"),
        # B STAYS above the centre dock door and is re-aimed, not moved, and the
        # reason is a gate rather than a preference. Sliding it west to the mouth
        # of aisle A2|A3 does take that aisle from 54 % to 100 % single-camera
        # cover -- but it drops the worst of the three diverse primary routes
        # from 0.553 to 0.465 two-camera fraction, under the 0.50 the fusion
        # study needs (camera_layout_decision.json). Every aim and every x from
        # -3.35 to -2.50 was measured; none recovers it, because the loss comes
        # from B's POSITION, not its aim. Re-aiming in place is free by contrast:
        # +115 deg instead of +96 buys whole-floor two-camera cover 44.4 -> 49.2 %
        # and floor cover 86.9 -> 87.6 % with the route fraction untouched.
        #
        # The cost, stated plainly: aisle A2|A3 stays near 54 %. Covering it and
        # holding the route gate at the same time is not possible with five
        # mounts on this geometry -- it needs a sixth.
        Camera("B", -1.50, -9.72, 5.00, 115.0, 48.0,
               "south wall above the centre dock door, across the apron and up "
               "into section A", "#eb6834"),
        # C is a north-wall mount at the head of aisle A1|A2. Two separate
        # measurements set it, both on this geometry and both stock states:
        #
        #  * POSITION. From the north-west corner it saw 17.8 % of aisle A1|A2,
        #    and re-aiming the corner mount does not help at all -- A1|A2 stays
        #    at 17.8 % at every yaw from -30 to -70 deg. The aisle head is the
        #    only place on the wall that fixes it.
        #  * AIM. Looking straight DOWN the aisle at -90 deg is the trap. It
        #    buys the last 1.7 points of that aisle (98.3 -> 100 %) and pays for
        #    them with the rest of the building: whole-floor cover 86.8 -> 82.0,
        #    two-camera cover 45.2 -> 42.5, and this camera's own share of the
        #    floor 18.6 -> 10.9 %, i.e. it stops being a general-purpose camera
        #    and becomes an aisle periscope. At -60 deg it sees the aisle AND
        #    the floor either side of it.
        Camera("C", -6.95, 9.45, 5.00, -60.0, 38.0,
               "north wall at the head of aisle A1|A2, angled across section A",
               "#1baf7a"),
        Camera("D", 11.45, 7.20, 5.00, -140.0, 38.0,
               "east wall in the cross aisle, over the box rows", "#4a3aa7"),
        Camera("E", 11.45, -9.45, 5.00, 132.0, 42.0,
               "south-east corner, over the block stacks", "#e34948"),
    ]
    # Chosen by search, not by taste: 918 wall mounts (four walls, this 5.00 m
    # cap, 9 yaws x 3 pitches each) were ray-cast against both stock states and
    # 5-camera sets scored by greedy seeding plus single-slot swaps. See
    # camera_sweep.py and camera_sweep.json. Worse-of-both-states result for the
    # set above, on drivable cells:
    #
    #   whole floor      87.6 % single-camera, 49.2 % two-camera
    #   aisle A1|A2      98.3 %      aisle A2|A3      54.1 %
    #   C block aisle 1  29.0 %      C block aisle 2  32.5 %
    #   smallest share of the floor held by any one camera: 18.6 %
    #   worst primary route: 0.553 two-camera fraction (gate 0.50) -- PASSES
    #
    # KNOWN RESIDUAL, and it is the honest weak point of this set: the two
    # section-C block aisles stay near 30 %. They are the worst-covered driveable
    # ground in the building and always were -- the rack aisles simply got the
    # attention first. No 5-camera set in the pool covers both halves: the best
    # "nothing is blind" set found (all four aisles >= 98 %) does it by putting
    # three cameras on the east wall, which leaves 31.2 % of the floor west of
    # the artery blind against 8.0 % here. Covering both needs a sixth mount.
    L.notes = [
        "Section A: 3 back-to-back pairs, 2.00 m picking aisles, runs 11.75 m "
        "(three native modules), 3-level racking to 4.2 m.",
        "Section B: one pair turned 90 deg against the north wall, runs 7.83 m "
        "(two modules), top beam 2.61 m -- a different product family, not a "
        "copy of A.",
        "Section C: no racking. Three long floor-storage rows with two 1.80 m "
        "aisles running their full 8.05 m length, three stacks per row at one, "
        "two or three pallets high: 1.06 / 2.12 / 3.17 m.",
    ]
    bad = L.check()
    if bad:
        raise AssertionError(f"contents escape their zone: {bad[:5]}")
    return L


if __name__ == "__main__":
    L = build()
    print(f"{len(L.zones)} zones | {len(L.fill_a)} objects at peak | {len(L.fill_b)} after")
    print(f"apron        {SITE[2]:+.2f} .. {APRON_TOP:+.2f}  ({APRON_TOP - SITE[2]:.2f} m deep)")
    print(f"section A    x {A_X[0]-PAIR/2:+.2f} .. {A_X[-1]+PAIR/2:+.2f}, "
          f"y {A_Y0:+.2f} .. {A_Y1:+.2f}   aisles {A_X[1]-A_X[0]-PAIR:.2f} m")
    print(f"artery       x {ARTERY[0]:+.2f} .. {ARTERY[1]:+.2f}  ({ARTERY[1]-ARTERY[0]:.2f} m)")
    print(f"section B    x {B_X0:+.2f} .. {B_X1:+.2f}, y {B_Y0:+.2f} .. {B_Y1:+.2f}   "
          f"{len(B_Y)} pair against the north wall")
    print("section C    three long floor-storage rows")
    for name, y0, y1 in C_ROWS:
        print(f"   {name:<3} x {C_X0:+.2f}..{C_X1:+.2f} ({C_X1-C_X0:.2f} m long)  "
              f"y {y0:+.2f}..{y1:+.2f} ({y1-y0:.2f} m deep)  stacks {C_PEAK[name]} high")
    for a, b in C_AISLES:
        print(f"   box aisle y {a:+.2f}..{b:+.2f}: {b-a:.2f} m wide, {C_X1-C_X0:.2f} m long")
    print(f"   east cross aisle y {CROSS_E[0]:+.2f}..{CROSS_E[1]:+.2f}: "
          f"{CROSS_E[1]-CROSS_E[0]:.2f} m")
    print(f"north aisle  west {SITE[3]-A_Y1:.2f} m, east cross aisle "
          f"{CROSS_E[1]-CROSS_E[0]:.2f} m")
