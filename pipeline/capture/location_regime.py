#!/usr/bin/env python3
"""Build and verify the pre-capture spatial regime for the final thesis pipeline.

This lock is deliberately geometry-only.  It allocates whole 3.2 m spatial
blocks inside one immutable master capture before looking at an RGB image or
detector output.  Every camera, heading, repetition and augmentation derived
from one ground position inherits the same role.

The generated CSV is the only legal stationary-pose index for the final
pipeline.  Existing datasets remain historical diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GENERATED = HERE / "generated"
POSITION_CSV = GENERATED / "camera_capture_positions_v3.csv"
POSE_FILE = GENERATED / "camera_capture_poses_v3.json"
MANIFEST = HERE / "capture_map_manifest.json"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
WORLD_PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
TASKS = REPO / "src/experiments/config/tasks.yaml"
WORLD_MANIFEST = REPO / "world/world_freeze_manifest.json"
ROBOT_MANIFEST = HERE / "robot_target_manifest.json"

sys.path.insert(0, str(REPO / "world"))
import route_tasks as rt  # noqa: E402
from coverage import grid as coverage_grid  # noqa: E402
from coverage import height_map, make_cam, visible_from  # noqa: E402
from warehouse_v2 import build as build_layout  # noqa: E402


LOCK_ID = "WHV2-CAMERA-CAPTURE-MAP-V3"
CAMERA_IDS = tuple(f"camera_{letter}" for letter in "ABCDE")
CAMERA_MODELS = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
    "camera_E": "external_camera_e",
}
CANDIDATE_LATTICE_SPACING_M = 0.4
BLOCK_SIZE_M = 3.2
TARGET_POSITION_COUNT = 400
HEADINGS_PER_POSITION = 8
HEADING_STEP_DEG = 360.0 / HEADINGS_PER_POSITION
HEADING_PHASE_SEED = 20260909
X_VALUES = np.arange(-10.8, 10.8 + 1.0e-9, CANDIDATE_LATTICE_SPACING_M)
Y_VALUES = np.arange(-8.8, 8.8 + 1.0e-9, CANDIDATE_LATTICE_SPACING_M)

# These whole-block assignments were selected by a geometry-only balance
# search.  The objective matches each role's position fraction and every
# camera's line-of-sight fraction.  No RGB, semantic mask, detector confidence
# or localization residual enters the assignment.
MASTER_ROLE_BLOCKS = {
    "final_audit": ((0, 1), (4, 5), (5, 0), (7, 2)),
    "commissioning_fit": (
        (0, 0), (0, 2), (0, 5), (1, 0), (1, 1), (1, 3), (1, 5),
        (2, 0), (2, 3), (2, 4), (2, 5), (3, 0), (3, 5), (4, 0),
        (4, 4), (5, 1), (5, 3), (6, 0), (6, 1), (6, 2), (6, 3),
        (6, 5), (7, 1), (7, 5),
    ),
    "detector_validation": ((0, 3), (1, 4), (2, 1), (4, 3), (7, 4)),
}

ROUTE_NAMES = (
    "fusion_network_traverse",
    "fusion_overlap_rich",
    "fusion_overlap_sparse",
    "fusion_long_traverse",
)
ROUTE_FILES = {
    name: REPO / f"world/routes/{name}.json"
    for name in ROUTE_NAMES
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def block_id(x: float, y: float) -> tuple[int, int]:
    return math.floor((x + 12.0) / BLOCK_SIZE_M), math.floor((y + 9.6) / BLOCK_SIZE_M)


def master_roles(blocks: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    assigned: dict[tuple[int, int], str] = {}
    for role, role_blocks in MASTER_ROLE_BLOCKS.items():
        for item in role_blocks:
            if item in assigned:
                raise RuntimeError(f"Spatial block {item} assigned twice")
            assigned[item] = role
    unknown = set(assigned) - set(blocks)
    if unknown:
        raise RuntimeError(f"Assigned blocks are absent from the capture lattice: {sorted(unknown)}")
    for item in blocks:
        assigned.setdefault(item, "detector_fit")
    return assigned


def camera_contract() -> tuple[dict[str, dict], dict[str, tuple[float, ...]]]:
    world = ET.parse(WORLD).getroot().find("world")
    if world is None:
        raise RuntimeError(f"No world element in {WORLD}")
    include_by_name = {
        item.findtext("name", "").strip(): item for item in world.findall("include")
    }
    records: dict[str, dict] = {}
    poses: dict[str, tuple[float, ...]] = {}
    for camera_id, model_name in CAMERA_MODELS.items():
        include = include_by_name[model_name]
        pose = tuple(float(value) for value in include.findtext("pose", "").split())
        model_path = REPO / f"src/sim/models/{model_name}/model.sdf"
        model = ET.parse(model_path).getroot()
        sensor = model.find(".//sensor[@name='camera']")
        if sensor is None:
            raise RuntimeError(f"RGB sensor missing in {model_path}")
        camera = sensor.find("camera")
        if camera is None:
            raise RuntimeError(f"camera element missing in {model_path}")
        noise = camera.find("noise")
        records[camera_id] = {
            "model_name": model_name,
            "model_path": relative(model_path),
            "model_sha256": sha256_file(model_path),
            "pose_xyz_rpy_rad": list(pose),
            "image_topic": "/" + sensor.findtext("topic", "").strip(),
            "image_width_px": int(camera.findtext("image/width", "0")),
            "image_height_px": int(camera.findtext("image/height", "0")),
            "horizontal_fov_rad": float(camera.findtext("horizontal_fov", "nan")),
            "update_rate_hz": float(sensor.findtext("update_rate", "nan")),
            "rgb_noise_model": "none" if noise is None else noise.findtext("type", "unknown"),
        }
        poses[camera_id] = pose
    return records, poses


def stationary_positions() -> tuple[list[dict], dict[str, np.ndarray]]:
    map_x, map_y, driveable, _ = rt.driveable()
    profiles = yaml.safe_load(WORLD_PROFILE.read_text(encoding="utf-8"))
    profile = profiles["worlds"][WORLD.name]
    known_regions = list(profile.get("known_2d_regions") or [])
    traversable_regions = [
        region for region in known_regions
        if str(region.get("type", "")).strip().lower() == "traversable"
    ]
    excluded_regions = [
        region for region in known_regions
        if str(region.get("type", "")).strip().lower()
        not in {"traversable", "site_boundary"}
    ]

    def capture_legal(x: float, y: float) -> bool:
        inside_lane = any(
            float(region["xmin"]) + 0.05 <= x <= float(region["xmax"]) - 0.05
            and float(region["ymin"]) + 0.05 <= y <= float(region["ymax"]) - 0.05
            for region in traversable_regions
        )
        inside_exclusion = any(
            float(region["xmin"]) - 0.25 <= x <= float(region["xmax"]) + 0.25
            and float(region["ymin"]) - 0.25 <= y <= float(region["ymax"]) + 0.25
            for region in excluded_regions
        )
        return inside_lane and not inside_exclusion
    layout = build_layout()
    cov_x, cov_y = coverage_grid()
    height = height_map(layout, "A", cov_x, cov_y)
    visibility = {
        f"camera_{camera.name}": visible_from(make_cam(camera), height, cov_x, cov_y)
        for camera in layout.cameras
    }
    candidates = []
    for y in Y_VALUES:
        for x in X_VALUES:
            map_ix = int(round((float(x) - float(map_x[0])) / rt.RES))
            map_iy = int(round((float(y) - float(map_y[0])) / rt.RES))
            if not (0 <= map_ix < len(map_x) and 0 <= map_iy < len(map_y)):
                continue
            if not bool(driveable[map_iy, map_ix]):
                continue
            if not capture_legal(float(x), float(y)):
                continue
            cov_ix = int(round((float(x) - float(cov_x[0])) / 0.10))
            cov_iy = int(round((float(y) - float(cov_y[0])) / 0.10))
            visible = tuple(
                camera_id for camera_id in CAMERA_IDS
                if bool(visibility[camera_id][cov_iy, cov_ix])
            )
            candidates.append({
                "x_m": round(float(x), 3),
                "y_m": round(float(y), 3),
                "block": block_id(float(x), float(y)),
                "geometry_visible_camera_ids": visible,
            })
    if len(candidates) < TARGET_POSITION_COUNT:
        raise RuntimeError(
            f"Only {len(candidates)} legal candidate positions for target {TARGET_POSITION_COUNT}"
        )
    # Deterministic maximin thinning prioritizes spatial support.  Starting at the
    # southwest-most legal point fixes the otherwise arbitrary first point.
    points = np.asarray([(row["x_m"], row["y_m"]) for row in candidates], dtype=float)
    first = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    selected = [first]
    nearest_sq = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, TARGET_POSITION_COUNT):
        index = int(np.argmax(nearest_sq))
        selected.append(index)
        nearest_sq = np.minimum(
            nearest_sq, np.sum((points - points[index]) ** 2, axis=1)
        )
    positions = [candidates[index] for index in selected]
    positions.sort(key=lambda row: (row["y_m"], row["x_m"]))
    return positions, {"x": map_x, "y": map_y, "driveable": driveable}


def build_plan() -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    positions, map_data = stationary_positions()
    blocks = sorted({row["block"] for row in positions})
    roles = master_roles(blocks)
    phase_rng = np.random.default_rng(HEADING_PHASE_SEED)
    phase_by_block = {
        block: round(float(phase_rng.uniform(0.0, HEADING_STEP_DEG)), 6)
        for block in blocks
    }
    rows = []
    for index, row in enumerate(positions):
        bx, by = row["block"]
        phase = phase_by_block[row["block"]]
        headings = [round((phase + i * HEADING_STEP_DEG) % 360.0, 6)
                    for i in range(HEADINGS_PER_POSITION)]
        rows.append({
            "position_id": f"P{index:04d}",
            "x_m": row["x_m"],
            "y_m": row["y_m"],
            "block_id": f"B{bx:02d}_{by:02d}",
            "master_role": roles[row["block"]],
            "heading_phase_degrees": phase,
            "heading_degrees": ";".join(f"{value:.6f}" for value in headings),
            "camera_ids": ";".join(CAMERA_IDS),
            "geometry_visible_camera_ids": ";".join(row["geometry_visible_camera_ids"]),
        })

    role_counts = Counter(row["master_role"] for row in rows)
    visibility_counts: dict[str, dict[str, int]] = {}
    for camera_id in CAMERA_IDS:
        visibility_counts[camera_id] = {
            role: sum(
                camera_id in row["geometry_visible_camera_ids"].split(";")
                for row in rows if row["master_role"] == role
            )
            for role in ("detector_fit", "detector_validation", "commissioning_fit", "final_audit")
        }

    summary = {
        "position_count": len(rows),
        "spatial_block_count": len(blocks),
        "heading_count_per_position": HEADINGS_PER_POSITION,
        "camera_count_per_pose": len(CAMERA_IDS),
        "robot_pose_count": len(rows) * HEADINGS_PER_POSITION,
        "attempted_views_per_full_capture": len(rows) * HEADINGS_PER_POSITION * len(CAMERA_IDS),
        "master_role_position_counts": dict(sorted(role_counts.items())),
        "geometry_visible_position_counts_by_camera_and_role": visibility_counts,
    }
    return rows, summary, map_data


def write_csv(rows: list[dict]) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    with POSITION_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_pose_file(rows: list[dict]) -> None:
    poses = []
    for numeric_position_id, row in enumerate(rows):
        headings = [float(value) for value in row["heading_degrees"].split(";")]
        for heading_id, heading_deg in enumerate(headings):
            poses.append({
                "position_id": numeric_position_id,
                "position_key": row["position_id"],
                "heading_id": heading_id,
                "x_idx": numeric_position_id,
                "y_idx": numeric_position_id,
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                "yaw": math.radians(heading_deg),
                "heading_degrees": heading_deg,
                "block_id": row["block_id"],
                "stratum": row["master_role"],
            })
    payload = {
        "schema": "thesis_master_capture_pose_plan.v1",
        "lock_id": LOCK_ID,
        "position_count": len(rows),
        "headings_per_position": HEADINGS_PER_POSITION,
        "camera_count": len(CAMERA_IDS),
        "planned_camera_views": len(poses) * len(CAMERA_IDS),
        "poses": poses,
    }
    POSE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def route_polylines() -> dict[str, list[list[float]]]:
    routes = {}
    for name, path in ROUTE_FILES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        route = payload.get("route") or payload.get("points") or payload.get("polyline")
        if route is None and isinstance(payload.get("polyline_canonical_json"), str):
            route = json.loads(payload["polyline_canonical_json"])
        if route is None:
            for value in payload.values():
                if isinstance(value, list) and value and isinstance(value[0], list):
                    route = value
                    break
        if not isinstance(route, list):
            raise RuntimeError(f"Cannot locate route polyline in {path}")
        routes[name] = [[float(point[0]), float(point[1])] for point in route]
    return routes


def build_manifest(rows: list[dict], summary: dict) -> dict:
    cameras, _ = camera_contract()
    route_data = route_polylines()
    world_lock = json.loads(WORLD_MANIFEST.read_text(encoding="utf-8"))
    robot_lock = json.loads(ROBOT_MANIFEST.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "lock_id": LOCK_ID,
        "status": "locked_pre_capture",
        "evidence_boundary": "camera_sensor_and_spatial_capture_design_only_no_rgb_or_detector_output",
        "upstream": {
            "world_lock_id": world_lock["freeze_id"],
            "world_manifest": relative(WORLD_MANIFEST),
            "world_manifest_sha256": sha256_file(WORLD_MANIFEST),
            "robot_lock_id": robot_lock["lock_id"],
            "robot_manifest": relative(ROBOT_MANIFEST),
            "robot_manifest_sha256": sha256_file(ROBOT_MANIFEST),
        },
        "camera_contract": cameras,
        "spatial_design": {
            "world": relative(WORLD),
            "world_sha256": sha256_file(WORLD),
            "candidate_lattice_spacing_m": CANDIDATE_LATTICE_SPACING_M,
            "position_selection": "deterministic maximin thinning to 400 legal positions",
            "spatial_block_size_m": BLOCK_SIZE_M,
            "heading_design": {
                "count_per_position": HEADINGS_PER_POSITION,
                "step_degrees": HEADING_STEP_DEG,
                "phase_rule": "one seeded U[0,45 degree) phase per 3.2 m spatial block",
                "phase_seed": HEADING_PHASE_SEED,
            },
            "camera_ids": list(CAMERA_IDS),
            "position_index": relative(POSITION_CSV),
            "position_index_sha256": sha256_file(POSITION_CSV),
            "capture_pose_file": relative(POSE_FILE),
            "capture_pose_file_sha256": sha256_file(POSE_FILE),
            "master_split": {
                "roles": ["detector_fit", "detector_validation", "commissioning_fit", "final_audit"],
                "role_blocks": {
                    role: [list(item) for item in role_blocks]
                    for role, role_blocks in MASTER_ROLE_BLOCKS.items()
                },
                "rule": "one master capture; whole 3.2 m blocks; all cameras/headings/derived data from a position remain in one role; exact role sizes 80/40/240/40",
                "selection_inputs": "driveable geometry and geometric camera line-of-sight only",
            },
            "summary": summary,
        },
        "regime_contract": {
            "generic_pretraining": (
                "Only generic base weights or a separately frozen randomized-warehouse corpus; "
                "no frame from warehouse_v2 or its shipout pair."
            ),
            "fixed_camera_detector_fit": (
                "Use detector_fit only for gradients; detector_validation only for early stopping "
                "and checkpoint choice."
            ),
            "commissioning": (
                "After detector weights are immutable, run them on the untouched commissioning_fit images. "
                "Use those outputs for confidence/admission selection and downstream correction, R_hit and q "
                "fitting; final_audit stays sealed until the entire observation stack is frozen."
            ),
            "navigation": (
                "Navigation may operate in the same commissioned warehouse because the claim is in-situ "
                "commissioning, not unseen-building transfer. It must use new continuous drives, frozen "
                "tasks/arms/seeds, and no navigation outcome may alter any detector, gate or downstream model."
            ),
            "same_data_rule": (
                "One immutable master dataset supplies every fitted artifact, but an RGB frame, semantic label, "
                "pose repetition or derived crop may never cross a role boundary."
            ),
            "capture_content": (
                "Retain synchronized RGB and evaluation-only robot semantic mask for all 16,000 "
                "attempted views, including absent, fully occluded, sliver and border-truncated cases."
            ),
            "ideal_sensor_caveat": (
                "The frozen RGB sensors declare no stochastic pixel-noise model. Exact stationary repeats "
                "therefore do not count as independent noise samples; later R_hit work must use declared local "
                "pose/yaw perturbations or continuous drives and name that conditioning neighborhood."
            ),
        },
        "localization_usability_contract": {
            "detector_return_is_not_measurement": True,
            "training_label_policy": (
                "Stage 04 must retain every attempted view in the audit table, but only a semantic robot "
                "silhouette with sufficient projected-hull width/height support, ground-contact support and "
                "no forbidden border truncation may become a detector-fit positive."
            ),
            "runtime_policy": (
                "Stage 06 must freeze a gate using detector confidence plus observable box size, border state "
                "and agreement with the projected robot hull. Semantic masks and ground truth are evaluation-only."
            ),
            "camera_D_sliver_challenge": {
                "id": "D_SLIVER_001",
                "x_m": -3.34375,
                "y_m": 7.41111,
                "yaw_rad": math.pi,
                "camera_id": "camera_D",
                "historical_visible_rgb_difference_pixels": 102,
                "historical_detected_box_width_px": 19.872,
                "historical_detected_box_height_px": 26.120,
                "historical_projected_hull_width_px": 48.622,
                "required_final_outcome": "not_admitted_as_localization_measurement",
                "note": "Historical measurements locate the challenge only; they do not set a fitted threshold.",
            },
        },
        "navigation_location_regime": {
            "status": "candidate_routes_frozen_locations_only; arms/seeds remain stage_09",
            "route_names": list(ROUTE_NAMES),
            "route_files": {
                name: {"path": relative(ROUTE_FILES[name]), "sha256": sha256_file(ROUTE_FILES[name])}
                for name in ROUTE_NAMES
            },
            "route_geometry_sha256": canonical_sha256(route_data),
            "selection_rule": "geometry-predeclared routes only; no new detector or commissioning result may choose the final route suite",
        },
        "sources": {
            relative(Path(__file__)): sha256_file(Path(__file__)),
            relative(WORLD_PROFILE): sha256_file(WORLD_PROFILE),
            relative(TASKS): sha256_file(TASKS),
            relative(REPO / "world/route_tasks.py"): sha256_file(REPO / "world/route_tasks.py"),
            relative(REPO / "world/coverage.py"): sha256_file(REPO / "world/coverage.py"),
        },
    }


def draw_review(rows: list[dict], map_data: dict[str, np.ndarray], output: Path) -> None:
    colors = {
        "detector_fit": "#2f77b5",
        "detector_validation": "#e69f00",
        "final_audit": "#b23a48",
        "commissioning_fit": "#3a8f5b",
    }
    x, y, mask = map_data["x"], map_data["y"], map_data["driveable"]
    edge_x = np.concatenate([x - rt.RES / 2, [x[-1] + rt.RES / 2]])
    edge_y = np.concatenate([y - rt.RES / 2, [y[-1] + rt.RES / 2]])
    _camera_records, camera_poses = camera_contract()
    routes = route_polylines()

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.8), sharex=True, sharey=True)
    for ax in axes:
        ax.pcolormesh(
            edge_x, edge_y, np.ma.masked_where(~mask, mask.astype(float)),
            cmap=matplotlib.colors.ListedColormap(["#edf1f4"]), shading="flat", zorder=0,
        )
        ax.contour(x, y, mask.astype(float), levels=[0.5], colors=["#a8b2ba"], linewidths=0.7)
        for camera_id, pose in camera_poses.items():
            ax.plot(pose[0], pose[1], "^", color="#202830", ms=7, zorder=8)
            ax.text(pose[0], pose[1] + 0.42, camera_id[-1], ha="center", va="bottom", fontsize=8, weight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-12.2, 12.2)
        ax.set_ylim(-10.0, 10.0)
        ax.grid(alpha=0.12, linewidth=0.5)
        ax.set_xlabel("east x (m)")
    axes[0].set_ylabel("north y (m)")

    for role in ("detector_fit", "detector_validation", "commissioning_fit", "final_audit"):
        subset = [row for row in rows if row["master_role"] == role]
        axes[0].scatter([r["x_m"] for r in subset], [r["y_m"] for r in subset],
                        s=22, c=colors[role], edgecolors="white", linewidths=0.35, label=role.replace("_", " "))
    axes[0].plot(-3.34375, 7.41111, marker="*", ms=14, color="#d00000", mec="white", mew=0.7, zorder=10)
    axes[0].annotate("D sliver\nmandatory reject", xy=(-3.34375, 7.41111), xytext=(-0.8, 8.8),
                     fontsize=8, color="#9d0000", arrowprops=dict(arrowstyle="->", color="#9d0000"))
    axes[0].set_title("1  One immutable master dataset\nwhole 3.2 m blocks; four disjoint roles")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)

    for role in ("detector_fit", "detector_validation"):
        subset = [row for row in rows if row["master_role"] == role]
        axes[1].scatter([r["x_m"] for r in subset], [r["y_m"] for r in subset],
                        s=12, c="#c8cdd2", edgecolors="none", alpha=0.55)
    for role in ("commissioning_fit", "final_audit"):
        subset = [row for row in rows if row["master_role"] == role]
        axes[1].scatter([r["x_m"] for r in subset], [r["y_m"] for r in subset],
                        s=25, c=colors[role], edgecolors="white", linewidths=0.35, label=role.replace("_", " "))
    axes[1].set_title("2  Frozen YOLO predicts untouched images\nfit correction / R-hit / q; final audit sealed")
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)

    route_colors = ("#0072b2", "#009e73", "#d55e00", "#7b3294")
    for (name, points), color in zip(routes.items(), route_colors):
        arr = np.asarray(points)
        axes[2].plot(arr[:, 0], arr[:, 1], color=color, lw=2.1, alpha=0.9, label=name.replace("fusion_", ""))
    axes[2].set_title("3  Navigation locations locked now\nnew continuous drives only; arms/seeds lock at stage 09")
    axes[2].legend(loc="lower left", fontsize=7.5, frameon=False)

    fig.suptitle(
        "Final thesis spatial regime — same warehouse is allowed; the same evidence is not",
        fontsize=15, weight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.02,
        "Every dot represents 8 block-phased headings and all 5 camera attempts: 16,000 master views. No image or derived crop crosses a role boundary.",
        ha="center", fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def compare(expected: object, actual: object, path: str = "manifest") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [f"{path}: keys differ"]
        failures = []
        for key in expected:
            failures.extend(compare(expected[key], actual[key], f"{path}.{key}"))
        return failures
    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Create the initial lock artifacts.")
    parser.add_argument("--review-out", type=Path, default=None, help="Optional PNG review figure.")
    args = parser.parse_args()

    rows, summary, map_data = build_plan()
    if args.write:
        write_csv(rows)
        write_pose_file(rows)
        candidate = build_manifest(rows, summary)
        MANIFEST.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not POSITION_CSV.is_file() or not MANIFEST.is_file():
        raise RuntimeError("Capture map lock does not exist; run once with --write.")
    candidate = build_manifest(rows, summary)
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = compare(expected, candidate)
    if args.review_out is not None:
        draw_review(rows, map_data, args.review_out.expanduser().resolve())
    if failures:
        print("CAMERA CAPTURE MAP LOCK: FAIL")
        for failure in failures[:20]:
            print(f"  - {failure}")
        return 1
    print(f"CAMERA CAPTURE MAP LOCK: PASS {LOCK_ID}")
    print(f"  positions={summary['position_count']} blocks={summary['spatial_block_count']} attempted_views={summary['attempted_views_per_full_capture']}")
    print(f"  roles={summary['master_role_position_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
