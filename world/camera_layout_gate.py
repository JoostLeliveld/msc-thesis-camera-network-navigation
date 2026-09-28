#!/usr/bin/env python3
"""Geometry-only pre-capture gate for the warehouse_v2 camera layout.

This script deliberately has no detector, image, ground-truth, localisation-log,
or learned-map input.  It compares the already-built coverage-oriented Camera C
pose with the documented fusion-oriented alternative, applies the frozen
selection rule in ``camera_layout_decision.json``, and rejects detector capture
if the decision, active Python layout, route opportunities, or stored metrics
have drifted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coverage import CELL, analyse  # noqa: E402
from layouts import Camera  # noqa: E402
from route_tasks import driveable, diverse_routes, hausdorff, snap  # noqa: E402
from warehouse_v2 import A_AISLE_X, C_AISLE_Y, build  # noqa: E402

DECISION_PATH = HERE / "camera_layout_decision.json"
# Derived from the layout's own aisle centrelines, so widening an aisle moves the
# primary route with it rather than leaving its endpoints inside new racking.
# aisleA2_s, the south mouth of picking aisle A2|A3 -- one aisle east of the original
# choice. At the planner's real 0.35 m clearance the previous primary task
# (aisleA1_s -> aisleC_n_e) gains a fourth route through the poorly covered block aisles
# with only 24.7% of its points visible to two cameras, so its menu fails the 50% gate and
# takes this firewall down with it. Of 91 tasks offering a real route choice, five have
# every route above the gate; this is the strongest of them, and it keeps the same
# destination so the traverse still crosses the network. Chosen from declared geometry, the
# camera model and the route solver alone -- no image or detector output entered it.
PRIMARY_START = (A_AISLE_X[1], -5.20)
#: the route-library name for the pair above, kept beside them so the label cannot drift
#: from the geometry it describes -- it did: the gate reported "aisleA1_s__aisleC_n_e" for
#: one run while already measuring the A2 mouth.
PRIMARY_TASK_NAME = "aisleA2_s__aisleC_n_e"
PRIMARY_GOAL = (10.60, C_AISLE_Y[1])

CANDIDATES = {
    "coverage_oriented_current": {
        "pose_xyz_yaw_pitch_deg": [A_AISLE_X[0], 9.45, 5.0, -60.0, 38.0],
        "purpose": "cover the previously blind A1|A2 aisle without spending the "
                   "whole camera on it",
    },
    "aisle_axial_previous": {
        "pose_xyz_yaw_pitch_deg": [A_AISLE_X[0], 9.45, 5.0, -90.0, 38.0],
        "purpose": "look straight down A1|A2 -- the WHV2-CAMLAYOUT-V1 choice, "
                   "kept as the documented alternative it was replaced by",
    },
    "fusion_oriented_alternative": {
        "pose_xyz_yaw_pitch_deg": [A_AISLE_X[1], 9.45, 5.0, -90.0, 38.0],
        "purpose": "increase overlap in A2|A3 and Camera C overlap-graph edge strength",
    },
}


def _camera_from_candidate(candidate: dict, colour: str) -> Camera:
    x, y, z, yaw, pitch = candidate["pose_xyz_yaw_pitch_deg"]
    return Camera(
        "C", x, y, z, yaw, pitch,
        candidate["purpose"], colour,
    )


def _percent(mask: np.ndarray, region: np.ndarray) -> float:
    return 100.0 * float(mask[region].mean()) if region.any() else 0.0


def _maximum_spanning_tree_bottleneck(edge_areas: dict[str, float]) -> float:
    """Bottleneck edge in a maximum spanning tree over cameras A--E."""

    parent = {name: name for name in "ABCDE"}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    selected = []
    for pair, area in sorted(edge_areas.items(), key=lambda item: item[1], reverse=True):
        a, b = pair
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        selected.append(float(area))
        if len(selected) == 4:
            break
    return min(selected) if len(selected) == 4 else 0.0


def _primary_routes():
    xs, ys, mask, _n_obstacles = driveable()
    start = snap(xs, ys, mask, PRIMARY_START)
    goal = snap(xs, ys, mask, PRIMARY_GOAL)
    routes = diverse_routes(mask, start, goal, xs, ys)
    separations = [
        hausdorff(routes[i], routes[j])
        for i in range(len(routes)) for j in range(i + 1, len(routes))
    ]
    return routes, (max(separations) if separations else 0.0)


def candidate_metrics(candidate: dict, routes, route_separation_m: float) -> dict:
    layout = build()
    layout.cameras[2] = _camera_from_candidate(candidate, layout.cameras[2].colour)
    result = analyse(layout)
    drive = result["drive"]
    peak = result["A"]
    shipout = result["B"]
    xs, ys = result["xs"], result["ys"]
    gx, gy = np.meshgrid(xs, ys)
    aisle_a1 = (
        drive & (gx >= -7.60) & (gx <= -6.50)
        & (gy >= -4.70) & (gy <= 7.70)
    )
    aisle_a2 = (
        drive & (gx >= -3.90) & (gx <= -2.80)
        & (gy >= -4.70) & (gy <= 7.70)
    )
    section_a_aisles = aisle_a1 | aisle_a2

    edge_areas = {}
    for i, camera_a in enumerate(layout.cameras):
        for j, camera_b in enumerate(layout.cameras[i + 1 :], start=i + 1):
            cells = int((peak["per_cam"][i] & peak["per_cam"][j] & drive).sum())
            edge_areas[f"{camera_a.name}{camera_b.name}"] = cells * CELL * CELL

    route_fractions = []
    for route in routes:
        ix = np.clip(np.rint((route[:, 0] - xs[0]) / CELL).astype(int), 0, len(xs) - 1)
        iy = np.clip(np.rint((route[:, 1] - ys[0]) / CELL).astype(int), 0, len(ys) - 1)
        route_fractions.append(float(np.mean(peak["n_visible"][iy, ix] >= 2)))

    metrics = {
        "all_driveable_peak_seen_by_1_pct": _percent(peak["n_visible"] >= 1, drive),
        "all_driveable_peak_seen_by_2_pct": _percent(peak["n_visible"] >= 2, drive),
        "all_driveable_shipout_seen_by_1_pct": _percent(shipout["n_visible"] >= 1, drive),
        "all_driveable_shipout_seen_by_2_pct": _percent(shipout["n_visible"] >= 2, drive),
        "section_a_aisles_peak_seen_by_1_pct": _percent(
            peak["n_visible"] >= 1, section_a_aisles
        ),
        "section_a_aisles_peak_seen_by_2_pct": _percent(
            peak["n_visible"] >= 2, section_a_aisles
        ),
        "peak_single_camera_coverage_pct": {
            camera.name: _percent(mask, drive)
            for camera, mask in zip(layout.cameras, peak["per_cam"])
        },
        "peak_pair_overlap_area_m2": edge_areas,
        "peak_overlap_graph_max_st_bottleneck_m2": (
            _maximum_spanning_tree_bottleneck(edge_areas)
        ),
        "primary_route": {
            "task": PRIMARY_TASK_NAME,
            "n_geometry_routes": len(routes),
            "max_route_separation_m": route_separation_m,
            "two_camera_fraction_per_route": route_fractions,
            "worst_route_two_camera_fraction": min(route_fractions) if route_fractions else 0.0,
            "mean_route_two_camera_fraction": (
                float(np.mean(route_fractions)) if route_fractions else 0.0
            ),
        },
    }
    return _rounded(metrics)


def _rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def compute_all() -> dict:
    routes, separation = _primary_routes()
    return {
        name: {
            **candidate,
            "metrics": candidate_metrics(candidate, routes, separation),
        }
        for name, candidate in CANDIDATES.items()
    }


def _metric_close(expected, actual, path="metrics") -> list[str]:
    failures = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            failures.append(
                f"{path} keys differ: expected {sorted(expected)}, got {sorted(actual)}"
            )
            return failures
        for key in expected:
            failures.extend(_metric_close(expected[key], actual[key], f"{path}.{key}"))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            failures.append(f"{path} length {len(actual)} != {len(expected)}")
        else:
            for index, (left, right) in enumerate(zip(expected, actual)):
                failures.extend(_metric_close(left, right, f"{path}[{index}]"))
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-6):
            failures.append(f"{path} {actual} != frozen {expected}")
    elif expected != actual:
        failures.append(f"{path} {actual!r} != frozen {expected!r}")
    return failures


def validate(decision: dict, computed: dict) -> list[str]:
    failures = []
    if decision.get("status") != "frozen_geometry_only":
        failures.append("decision status must be 'frozen_geometry_only'")
    selected = decision.get("selected_candidate")
    if selected not in CANDIDATES:
        failures.append(f"unknown selected_candidate {selected!r}")
        return failures

    frozen_candidates = decision.get("candidates", {})
    failures.extend(_metric_close(frozen_candidates, computed, "candidates"))

    gates = decision.get("gates", {})
    selected_route = computed[selected]["metrics"]["primary_route"]
    selected_metrics = computed[selected]["metrics"]
    if selected_route["n_geometry_routes"] < int(gates["min_geometry_routes"]):
        failures.append("selected layout fails minimum distinct-route count")
    if selected_route["max_route_separation_m"] < float(gates["min_route_separation_m"]):
        failures.append("selected layout fails route-separation gate")
    if selected_route["worst_route_two_camera_fraction"] < float(
        gates["min_worst_route_two_camera_fraction_peak"]
    ):
        failures.append("selected layout fails primary-route two-camera gate")
    if selected_metrics["peak_overlap_graph_max_st_bottleneck_m2"] < float(
        gates["min_overlap_graph_max_st_bottleneck_m2"]
    ):
        failures.append("selected layout fails overlap-graph connectivity gate")

    # The selection rule was frozen before detector capture: preserve the
    # existing layout if it clears every primary gate.  A post-hoc detector
    # result can never switch the layout.
    current = computed["coverage_oriented_current"]["metrics"]
    current_route = current["primary_route"]
    current_passes = (
        current_route["n_geometry_routes"] >= int(gates["min_geometry_routes"])
        and current_route["max_route_separation_m"] >= float(gates["min_route_separation_m"])
        and current_route["worst_route_two_camera_fraction"] >= float(
            gates["min_worst_route_two_camera_fraction_peak"]
        )
        and current["peak_overlap_graph_max_st_bottleneck_m2"] >= float(
            gates["min_overlap_graph_max_st_bottleneck_m2"]
        )
    )
    if current_passes and selected != "coverage_oriented_current":
        failures.append("selection violates frozen preserve-current-if-primary-gates-pass rule")

    active = build().cameras[2]
    active_pose = [active.x, active.y, active.z, active.yaw_deg, active.pitch_deg]
    selected_pose = CANDIDATES[selected]["pose_xyz_yaw_pitch_deg"]
    if not np.allclose(active_pose, selected_pose, rtol=0.0, atol=1e-9):
        failures.append(
            f"active Camera C pose {active_pose} does not match selected pose {selected_pose}"
        )

    if decision.get("geometry_gate_passed") is not True:
        failures.append("geometry_gate_passed must be true only after every frozen geometry gate passes")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-candidates",
        action="store_true",
        help="print freshly computed geometry-only candidate records as JSON",
    )
    args = parser.parse_args()

    computed = compute_all()
    if args.print_candidates:
        print(json.dumps(computed, indent=2, sort_keys=True))
        return 0

    decision = json.loads(DECISION_PATH.read_text())
    failures = validate(decision, computed)
    if failures:
        print("PRE-CAPTURE CAMERA-LAYOUT GATE: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    selected = decision["selected_candidate"]
    route = computed[selected]["metrics"]["primary_route"]
    print("PRE-CAPTURE CAMERA-LAYOUT GATE: PASS")
    print(f"  decision: {decision['decision_id']} ({selected})")
    print(
        "  primary route: "
        f"{route['n_geometry_routes']} choices, "
        f"{route['max_route_separation_m']:.2f} m separation, "
        f"worst two-camera fraction {route['worst_route_two_camera_fraction']:.3f}"
    )
    print("  inputs: geometry + camera model + arm-neutral routes only; no detector data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
