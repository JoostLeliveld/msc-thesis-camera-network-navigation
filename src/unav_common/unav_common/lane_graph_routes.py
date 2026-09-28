"""Map-derived multistart route-seed generator (condition-neutral, no visibility).

Rationale
---------
The hierarchical planner solves the long-horizon global EFE route ONCE (it does
not replan the global route; see efe_agent_node._plan_once). The L-BFGS-B solve
is nonconvex, so it needs initial guesses that cover the distinct route basins
of the known map. This module generates those seeds by a fixed rule from the
DRIVEABLE map alone -- not hand-drawn, not derived from the GP/detector, and
identical across conditions.

Topology of the AWS world: horizontal cross-aisles connect the vertical rack
aisles.  More than one cross-aisle can be valid for the same start and goal
(notably the south loading apron and the lower main aisle).  Every valid
cross-aisle is a distinct optimizer basin worth offering.  Collapsing all
negative-y corridors to only the highest one hid the early south crossing used
by the hard routes and forced the robot to remain in the far-west aisle.

For each valid horizontal corridor we emit lane-centre Manhattan waypoints:

    W = [ (x_start, y_corridor), (x_goal, y_corridor), (x_goal, y_goal) ]

i.e. move along the start aisle to the corridor centre-line, run along the
corridor to the goal column, then up/down the goal aisle to the goal.

Every corridor centre-line and its usable x-extent is read from the driveable
geometry, and every emitted segment is validated to lie inside the driveable
union (so a corridor that does not actually connect start and goal is rejected).
"""
from __future__ import annotations

import json
import math
KEEPOUT_MODEL = 'keepout_region'

from typing import List, Tuple, Sequence
import numpy as np

from .occlusion_geometry import scene_from_json, signed_distance_to_union_xy
from .rectangular_footprint import RectangularFootprint

XY = Tuple[float, float]
SHORT_START_CONNECTOR_M = 0.10


def _horizontal_corridor_centres(prisms) -> List[float]:
    """Distinct centre-y of horizontal (wide) driveable prisms, low -> high."""
    ys = []
    for p in prisms:
        w = float(p.xmax - p.xmin)
        h = float(p.ymax - p.ymin)
        if w > h:  # horizontal corridor
            ys.append(round(0.5 * (p.ymin + p.ymax), 3))
    out: List[float] = []
    for y in sorted(set(ys)):
        if not out or abs(y - out[-1]) > 0.25:
            out.append(y)
    return out


def _vertical_corridor_centres(prisms) -> List[float]:
    """Distinct centre-x of vertical (tall) driveable prisms, west -> east."""
    xs = []
    for p in prisms:
        w = float(p.xmax - p.xmin)
        h = float(p.ymax - p.ymin)
        if h > w:
            xs.append(round(0.5 * (p.xmin + p.xmax), 3))
    out: List[float] = []
    for x in sorted(set(xs)):
        if not out or abs(x - out[-1]) > 0.25:
            out.append(x)
    return out


def _route_centres_from_geometry(
    driveable_geometry_json: str,
    prisms,
) -> tuple[List[float], List[float]]:
    """Return declared routing axes, falling back to rectangle inference.

    A driveable *support* can be an exact rectangular decomposition of free
    floor with holes around obstacles.  Those decomposition cells are not
    themselves semantic aisles, so inferring a route centre from every cell
    would create spurious candidates.  Optional top-level routing axes keep
    the lane topology separate from the exact keep-in support.  Historical
    geometry has no such metadata and retains the previous inference exactly.
    """
    try:
        payload = json.loads(driveable_geometry_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}

    def declared(key: str) -> List[float] | None:
        raw = payload.get(key)
        if raw is None:
            return None
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{key} must be a non-empty list")
        values = sorted({round(float(item), 3) for item in raw})
        if not all(np.isfinite(values)):
            raise ValueError(f"{key} must contain only finite numbers")
        return values

    horizontal = declared('route_horizontal_centres')
    vertical = declared('route_vertical_centres')
    # Inference reads the centres of DRIVEABLE rectangles, which is only
    # meaningful for a keep-in map. A keep-out payload must declare its axes, or
    # inference would place corridor centre-lines inside obstacles - silently
    # wrong routes rather than an error.
    if payload.get('model_name') == KEEPOUT_MODEL and (horizontal is None or vertical is None):
        raise ValueError(
            'route_horizontal_centres and route_vertical_centres must be declared for '
            f'{KEEPOUT_MODEL!r} geometry; corridor centres cannot be inferred from an '
            'obstacle (keep-out) map')
    return (
        horizontal if horizontal is not None else _horizontal_corridor_centres(prisms),
        vertical if vertical is not None else _vertical_corridor_centres(prisms),
    )


def _segment_free(prisms, a: XY, b: XY, *, keep_out: bool,
                  step: float = 0.10, tol: float = 1e-3) -> bool:
    """True iff the straight segment a->b is drivable under this map's semantics.

    keep_out: the prisms are non-traversable boxes (one per collision footprint,
    grown by a fixed margin) and a valid segment stays OUTSIDE their union.
    Otherwise the prisms are a keep-in driveable union and a valid segment stays
    INSIDE it. The keep-out test also measured 1.5x faster, because the obstacle
    union has simpler boundary than the lane union it replaced.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = max(2, int(np.ceil(float(np.hypot(*(b - a))) / step)) + 1)
    pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
    sd = np.asarray(signed_distance_to_union_xy(tuple(prisms), pts, keep_in=not keep_out), float)
    return bool(np.all(sd > tol)) if keep_out else bool(np.all(sd <= tol))


def _is_keepout(driveable_geometry_json: str) -> bool:
    """True when the payload is the obstacle (keep-out) map."""
    try:
        return json.loads(driveable_geometry_json).get('model_name') == KEEPOUT_MODEL
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _dedupe(points: Sequence[XY]) -> List[XY]:
    out: List[XY] = []
    for p in points:
        q = (float(p[0]), float(p[1]))
        if not out or float(np.hypot(q[0] - out[-1][0], q[1] - out[-1][1])) > 1e-6:
            out.append(q)
    return out


def _has_collinear_retracing(points: Sequence[XY]) -> bool:
    """Return whether a route reverses along the same straight segment.

    A corridor outside both endpoint ordinates can otherwise produce
    ``a -> b -> c`` on one vertical lane, with ``b`` beyond ``c``.  That seed
    drives down a lane and immediately retraces it.  It is not a distinct
    topological route and can be favoured spuriously by an information cost.
    """
    route = _dedupe(points)
    for a, b, c in zip(route, route[1:], route[2:]):
        ab = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        scale = max(1.0, float(np.linalg.norm(ab) * np.linalg.norm(bc)))
        if abs(cross) <= 1e-9 * scale and float(np.dot(ab, bc)) < 0.0:
            return True
    return False


def _route_for_corridor(
    prisms,
    vertical_centres: Sequence[float],
    start: XY,
    goal: XY,
    y_corridor: float,
    keep_out: bool = False,
) -> List[XY] | None:
    """Build a lane-centre route through one horizontal cross-aisle.

    The old three-leg rule kept the exact start/goal x coordinates for both
    vertical legs. That misses a valid route whenever a rack aisle is locally
    blocked: the robot must first move along its current cross-aisle to an
    adjacent north/south aisle. Candidate vertical lanes are still read only
    from the driveable map.
    """
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])

    start_lanes = sorted(
        {sx, *(float(x) for x in vertical_centres)},
        key=lambda x: (abs(x - sx), x),
    )
    goal_lanes = sorted(
        {gx, *(float(x) for x in vertical_centres)},
        key=lambda x: (abs(x - gx), x),
    )
    for start_x in start_lanes:
        start_ok = (
            _segment_free(prisms, start, (start_x, sy), keep_out=keep_out)
            and _segment_free(prisms, (start_x, sy), (start_x, y_corridor), keep_out=keep_out)
        )
        if not start_ok:
            continue
        for goal_x in goal_lanes:
            points = _dedupe([
                start,
                (start_x, sy),
                (start_x, y_corridor),
                (goal_x, y_corridor),
                (goal_x, gy),
                goal,
            ])
            if all(
                _segment_free(prisms, a, b, keep_out=keep_out)
                for a, b in zip(points, points[1:])
            ):
                if _has_collinear_retracing(points):
                    continue
                route = points[1:]
                # A tiny snap from the real start to a nearby lane centre can
                # force a turn-move-turn manoeuvre at the site boundary.  It is
                # only a graph discretisation artifact, not a distinct route
                # basin. Collapse it when the direct segment to the following
                # waypoint is geometry-free. This rule is map-only and is
                # therefore identical for every camera-network condition.
                if (len(route) >= 2
                        and float(np.linalg.norm(
                            np.asarray(route[0]) - np.asarray(start)
                        )) <= SHORT_START_CONNECTOR_M
                        and _segment_free(
                            prisms, start, route[1], keep_out=keep_out)):
                    route = route[1:]
                return route
    return None


def generate_route_seeds(
    driveable_geometry_json: str,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> List[dict]:
    """Return condition-neutral lane-graph route seeds for ``start -> goal``.

    Output matches optimizer_initial_routes_json: a list of
    {"name": str, "waypoints": [[x,y], ...]}. Every horizontal corridor whose
    lane-centre Manhattan legs stay inside the driveable union is emitted. The
    ordering is deterministic (low y to high y), and no visibility or
    measurement data enters candidate generation.
    """
    scene = scene_from_json(driveable_geometry_json)
    prisms = tuple(scene.prisms)
    start = (float(start_xy[0]), float(start_xy[1]))
    goal = (float(goal_xy[0]), float(goal_xy[1]))

    keep_out = _is_keepout(driveable_geometry_json)
    centres, vertical_centres = _route_centres_from_geometry(
        driveable_geometry_json, prisms,
    )
    if not centres:
        return []

    below = [y for y in centres if y < 0.0]
    centred = [y for y in centres if y == 0.0]
    above = [y for y in centres if y > 0.0]
    valid_below = [
        (y, route)
        for y in below
        if (route := _route_for_corridor(
            prisms, vertical_centres, start, goal, y, keep_out=keep_out
        )) is not None
    ]
    valid_above = [
        (y, route)
        for y in above
        if (route := _route_for_corridor(
            prisms, vertical_centres, start, goal, y, keep_out=keep_out
        )) is not None
    ]
    valid_centred = [
        (y, route)
        for y in centred
        if (route := _route_for_corridor(
            prisms, vertical_centres, start, goal, y, keep_out=keep_out
        )) is not None
    ]

    # Prefer cross-aisles that lie between the endpoint ordinates. If at least
    # one such route is feasible, a corridor beyond both endpoints is not a
    # distinct necessary detour: it first drives away from the goal and can be
    # exploited as an information-gathering excursion. Retain outside-band
    # corridors only when map geometry leaves no progressing alternative.
    all_valid = valid_below + valid_centred + valid_above
    y_low, y_high = sorted((start[1], goal[1]))
    progressing = [(y, route) for y, route in all_valid
                   if y_low - 1e-6 <= y <= y_high + 1e-6]
    if progressing:
        allowed = {round(float(y), 6) for y, _route in progressing}
        valid_below = [(y, route) for y, route in valid_below
                       if round(float(y), 6) in allowed]
        valid_centred = [(y, route) for y, route in valid_centred
                         if round(float(y), 6) in allowed]
        valid_above = [(y, route) for y, route in valid_above
                       if round(float(y), 6) in allowed]

    routes: List[dict] = []
    for index, (y, waypoints) in enumerate(valid_below):
        if len(valid_below) == 1 or index == len(valid_below) - 1:
            name = "below_main_aisle"
        elif index == 0:
            name = "below_south_cross_aisle"
        else:
            name = f"below_cross_aisle_{index + 1}"
        routes.append({
            "name": name,
            "waypoints": [list(w) for w in waypoints],
        })

    for _y, waypoints in valid_centred:
        routes.append({
            "name": "centre_cross_aisle",
            "waypoints": [list(w) for w in waypoints],
        })

    for index, (y, waypoints) in enumerate(valid_above):
        if y < 3.0:
            name = "above_connector"
        elif index == len(valid_above) - 1:
            name = "above_cross_aisle"
        else:
            name = f"above_cross_aisle_{index + 1}"
        routes.append({
            "name": name,
            "waypoints": [list(w) for w in waypoints],
        })

    return routes


def generate_route_seeds_json(driveable_geometry_json: str, start_xy, goal_xy) -> str:
    import json
    return json.dumps(generate_route_seeds(driveable_geometry_json, start_xy, goal_xy))


def _turn_drive_route_clearance(
    points: Sequence[XY], start_yaw: float,
    collision: RectangularFootprint, boundary: RectangularFootprint,
) -> float:
    """Exact clearance of a rotate-then-translate waypoint initialization."""
    points = _dedupe(points)
    if len(points) < 2:
        return -math.inf
    pose = np.asarray([points[0][0], points[0][1], float(start_yaw)], dtype=float)
    minimum = min(collision.clearance(pose), boundary.clearance(pose))
    for destination in points[1:]:
        delta = np.asarray(destination, dtype=float) - pose[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            continue
        target_yaw = math.atan2(float(delta[1]), float(delta[0]))
        yaw_delta = (target_yaw - pose[2] + math.pi) % (2.0 * math.pi) - math.pi
        turned = pose.copy(); turned[2] += yaw_delta
        minimum = min(
            minimum,
            collision.sweep_clearance(pose, turned, yaw_delta=yaw_delta),
            boundary.sweep_clearance(pose, turned, yaw_delta=yaw_delta),
        )
        arrived = np.asarray([destination[0], destination[1], turned[2]], dtype=float)
        minimum = min(
            minimum,
            collision.sweep_clearance(turned, arrived, yaw_delta=0.0),
            boundary.sweep_clearance(turned, arrived, yaw_delta=0.0),
        )
        pose = arrived
    return float(minimum)


def repair_route_seeds_for_footprint(
    seeds: Sequence[dict], collision_geometry_json: str,
    driveable_geometry_json: str, start_pose: Sequence[float],
    *, robot_length_m: float = 0.80, robot_width_m: float = 0.55,
    target_clearance_m: float = 0.02, lateral_step_m: float = 0.025,
    lateral_limit_m: float = 0.75,
) -> List[dict]:
    """Repair only footprint-invalid vertical columns by a minimal lateral shift.

    Route topology and the selected horizontal corridor remain unchanged.  The
    search uses physical geometry only, is deterministic, and is therefore
    common to every camera-network arm.  Already hard-valid seeds are preserved
    exactly; the small positive target only prevents a repaired seed from being
    left numerically tangent to hard geometry.  It is not a soft-cost rule.
    """
    start = np.asarray(start_pose, dtype=float)
    if start.shape != (3,) or not np.isfinite(start).all():
        raise ValueError('start_pose must be finite [x,y,yaw]')
    if not all(np.isfinite([target_clearance_m, lateral_step_m, lateral_limit_m])):
        raise ValueError('repair distances must be finite')
    if target_clearance_m < 0 or lateral_step_m <= 0 or lateral_limit_m < lateral_step_m:
        raise ValueError('invalid repair distances')
    collision_scene = scene_from_json(collision_geometry_json)
    boundary_scene = scene_from_json(driveable_geometry_json)
    collision = RectangularFootprint(
        collision_scene.prisms, robot_length_m, robot_width_m)
    boundary = RectangularFootprint(
        boundary_scene.prisms, robot_length_m, robot_width_m, keep_in=True)
    repaired = []
    offsets = []
    for magnitude in np.arange(lateral_step_m, lateral_limit_m + 1e-12,
                               lateral_step_m):
        offsets.extend((float(magnitude), -float(magnitude)))
    for seed in seeds:
        original = [(float(start[0]), float(start[1]))] + [
            (float(point[0]), float(point[1])) for point in seed['waypoints']]
        clearance = _turn_drive_route_clearance(
            original, float(start[2]), collision, boundary)
        if clearance >= 0.0:
            repaired.append({**seed, 'footprint_seed_clearance_m': clearance,
                             'geometry_repaired': False})
            continue
        candidates = []
        for index, (first, second) in enumerate(zip(original, original[1:])):
            if abs(first[0] - second[0]) > 1e-9 or abs(first[1] - second[1]) <= 1e-9:
                continue
            for offset in offsets:
                shifted_x = first[0] + offset
                variant = list(original[:index])
                if index == 0:
                    variant.append(original[0])
                variant.extend(((shifted_x, first[1]), (shifted_x, second[1])))
                variant.extend(original[index + 2:])
                if variant[-1] != original[-1]:
                    variant.append(original[-1])
                variant = _dedupe(variant)
                value = _turn_drive_route_clearance(
                    variant, float(start[2]), collision, boundary)
                if value >= target_clearance_m:
                    candidates.append((abs(offset), -value, shifted_x, variant))
                    break
        if not candidates:
            repaired.append({**seed, 'footprint_seed_clearance_m': clearance,
                             'geometry_repaired': False,
                             'geometry_repair_failed': True})
            continue
        _magnitude, negative_clearance, shifted_x, best = min(candidates)
        repaired.append({
            **seed,
            'waypoints': [list(point) for point in best[1:]],
            'footprint_seed_clearance_m': -negative_clearance,
            'geometry_repaired': True,
            'repaired_vertical_x_m': shifted_x,
        })
    return repaired


def _remove_collinear(points: Sequence[XY], *, tol: float = 1e-9) -> List[XY]:
    """Return the same polyline without redundant collinear interior vertices."""
    points = _dedupe(points)
    if len(points) <= 2:
        return points
    out = [points[0]]
    for index in range(1, len(points) - 1):
        previous = np.asarray(out[-1], dtype=float)
        current = np.asarray(points[index], dtype=float)
        following = np.asarray(points[index + 1], dtype=float)
        first = current - previous
        second = following - current
        cross = float(first[0] * second[1] - first[1] * second[0])
        if abs(cross) <= tol and float(first @ second) >= -tol:
            continue
        out.append(points[index])
    out.append(points[-1])
    return _dedupe(out)


def _has_immediate_backtrack(points: Sequence[XY], *, tol: float = 1e-9) -> bool:
    """Reject a polyline that reverses along the segment it just traversed."""
    points = _dedupe(points)
    for index in range(1, len(points) - 1):
        first = np.asarray(points[index], dtype=float) - np.asarray(
            points[index - 1], dtype=float
        )
        second = np.asarray(points[index + 1], dtype=float) - np.asarray(
            points[index], dtype=float
        )
        cross = float(first[0] * second[1] - first[1] * second[0])
        if abs(cross) <= tol and float(first @ second) < -tol:
            return True
    return False


def generate_diverse_route_candidates(
    driveable_geometry_json: str,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    *,
    max_routes: int = 8,
) -> List[dict]:
    """Generate a bounded, diverse lane-graph set for finite route selection.

    Unlike :func:`generate_route_seeds`, which supplies one initializer per
    horizontal corridor, this enumerates valid start-lane/goal-lane variants.
    It first retains one distinct shortest route per cross-aisle, then fills the
    remaining budget by length.  Candidate generation uses geometry only and is
    therefore identical across experimental arms.
    """
    if isinstance(max_routes, bool) or int(max_routes) != max_routes or max_routes < 1:
        raise ValueError('max_routes must be a positive integer')
    scene = scene_from_json(driveable_geometry_json)
    prisms = tuple(scene.prisms)
    start = (float(start_xy[0]), float(start_xy[1]))
    goal = (float(goal_xy[0]), float(goal_xy[1]))
    keep_out = _is_keepout(driveable_geometry_json)
    horizontal, vertical = _route_centres_from_geometry(
        driveable_geometry_json, prisms,
    )

    by_corridor: List[tuple[float, List[dict]]] = []
    for corridor_y in horizontal:
        variants = []
        for start_x in sorted({start[0], *(float(x) for x in vertical)}):
            if not (
                _segment_free(prisms, start, (start_x, start[1]), keep_out=keep_out)
                and _segment_free(
                    prisms, (start_x, start[1]), (start_x, corridor_y), keep_out=keep_out
                )
            ):
                continue
            for goal_x in sorted({goal[0], *(float(x) for x in vertical)}):
                raw_full = _dedupe([
                    start,
                    (start_x, start[1]),
                    (start_x, corridor_y),
                    (goal_x, corridor_y),
                    (goal_x, goal[1]),
                    goal,
                ])
                if _has_immediate_backtrack(raw_full):
                    continue
                full = _remove_collinear(raw_full)
                if not all(
                    _segment_free(prisms, a, b, keep_out=keep_out)
                    for a, b in zip(full, full[1:])
                ):
                    continue
                length = float(sum(
                    np.linalg.norm(np.asarray(b) - np.asarray(a))
                    for a, b in zip(full, full[1:])
                ))
                variants.append({
                    'corridor_y': float(corridor_y),
                    'start_lane_x': float(start_x),
                    'goal_lane_x': float(goal_x),
                    'length_m': length,
                    'waypoints': [list(point) for point in full[1:]],
                })
        unique = {}
        for variant in sorted(
            variants,
            key=lambda item: (
                item['length_m'], item['start_lane_x'], item['goal_lane_x'],
                tuple(map(tuple, item['waypoints'])),
            ),
        ):
            key = tuple(map(tuple, variant['waypoints']))
            unique.setdefault(key, variant)
        by_corridor.append((float(corridor_y), list(unique.values())))

    selected = []
    selected_keys = set()
    # Preserve cross-aisle diversity before spending the remaining budget on
    # near-shortest lane variants.
    for _corridor_y, variants in by_corridor:
        for variant in variants:
            key = tuple(map(tuple, variant['waypoints']))
            if key not in selected_keys:
                selected.append(variant)
                selected_keys.add(key)
                break
        if len(selected) >= max_routes:
            break
    remaining = sorted(
        (variant for _y, variants in by_corridor for variant in variants
         if tuple(map(tuple, variant['waypoints'])) not in selected_keys),
        key=lambda item: (
            item['length_m'], item['corridor_y'], item['start_lane_x'],
            item['goal_lane_x'], tuple(map(tuple, item['waypoints'])),
        ),
    )
    for variant in remaining:
        if len(selected) >= max_routes:
            break
        key = tuple(map(tuple, variant['waypoints']))
        if key in selected_keys:
            continue
        selected.append(variant)
        selected_keys.add(key)

    selected.sort(key=lambda item: (
        item['length_m'], item['corridor_y'], item['start_lane_x'],
        item['goal_lane_x'], tuple(map(tuple, item['waypoints'])),
    ))
    routes = []
    for index, variant in enumerate(selected):
        routes.append({
            'name': (
                f"route_{index + 1:02d}_y{variant['corridor_y']:+.2f}"
                f"_sx{variant['start_lane_x']:+.2f}_gx{variant['goal_lane_x']:+.2f}"
            ),
            'waypoints': variant['waypoints'],
            'length_m': variant['length_m'],
            'corridor_y': variant['corridor_y'],
            'start_lane_x': variant['start_lane_x'],
            'goal_lane_x': variant['goal_lane_x'],
        })
    return routes
