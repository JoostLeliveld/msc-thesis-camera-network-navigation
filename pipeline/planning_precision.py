#!/usr/bin/env python3
"""Export planner precision fields directly from the frozen R0, R1, and R2 models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src/reliability"), str(REPO / "src/unav_common"), str(REPO)]
from reliability.projection import camera_model_from_world  # noqa: E402

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CAMERA_INCLUDES = tuple(f"external_camera{suffix}" for suffix in ("", "_b", "_c", "_d", "_e"))
GRID_STEP_M = 0.20
GRID_BOUNDS_M = (-10.8, 10.8, -8.8, 8.8)
EIGENVALUE_FLOOR_M2 = 1.0e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis(lower: float, upper: float, step: float) -> np.ndarray:
    return np.linspace(lower, upper, int(math.ceil((upper - lower) / step)) + 1)


def psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (vectors * np.maximum(values, EIGENVALUE_FLOOR_M2)) @ vectors.T


def ray_basis(camera_xy: np.ndarray, point: np.ndarray) -> np.ndarray:
    along = point - camera_xy
    norm = float(np.linalg.norm(along))
    if norm <= 1.0e-12:
        raise ValueError("planning grid contains a camera centre")
    along /= norm
    return np.column_stack((along, np.asarray([-along[1], along[0]])))


def spatial_ray_covariance(model: dict[str, np.ndarray], camera: str,
                           point: np.ndarray) -> tuple[np.ndarray, float]:
    local = model["spatial_camera"].astype(str) == camera
    points = model["spatial_reference_xy_m"][local]
    moments = model["spatial_second_moment_ray_m2"][local]
    distance2 = np.sum((points - point) ** 2, axis=1)
    count = min(int(model["k_neighbors"][0]), len(points))
    selected = np.argpartition(distance2, count - 1)[:count]
    length = float(model["length_scale_m"][0])
    weight = np.exp(-0.5 * distance2[selected] / length ** 2)
    support = float(weight.sum())
    prior_strength = float(model["prior_strength"][0])
    scatter = np.einsum("n,nij->ij", weight, moments[selected])
    covariance = (
        scatter + prior_strength * model["prior_covariance_ray_m2"]
    ) / (support + prior_strength)
    return psd(covariance), support


def write_artifact(path: Path, xs: np.ndarray, ys: np.ndarray,
                   precision: np.ndarray, support: np.ndarray, metadata: dict) -> None:
    np.savez_compressed(
        path,
        xs=xs,
        ys=ys,
        camera_ids=np.asarray(CAMERAS),
        matched_precision_m2_inv=precision,
        residual_support=support,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariance-models", required=True, type=Path)
    parser.add_argument(
        "--world", type=Path,
        default=REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    model_path = args.covariance_models.resolve()
    world = args.world.resolve()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    with np.load(model_path, allow_pickle=False) as source:
        model = {name: np.asarray(source[name]) for name in source.files}
    required = {
        "camera_order", "prior_covariance_ray_m2", "prior_strength",
        "global_covariance_ray_m2", "per_camera_covariance_ray_m2",
        "spatial_reference_xy_m", "spatial_second_moment_ray_m2",
        "spatial_camera", "k_neighbors", "length_scale_m",
    }
    missing = required - set(model)
    if missing:
        raise ValueError(f"covariance artifact lacks {sorted(missing)}")
    order = tuple(model["camera_order"].astype(str))
    if order != CAMERAS:
        raise ValueError("covariance artifact camera order is not canonical")

    camera_xy = {
        camera: np.asarray(camera_model_from_world(
            world, include_name=include).cam_pos[:2], dtype=float)
        for camera, include in zip(CAMERAS, CAMERA_INCLUDES, strict=True)
    }
    xs = axis(GRID_BOUNDS_M[0], GRID_BOUNDS_M[1], GRID_STEP_M)
    ys = axis(GRID_BOUNDS_M[2], GRID_BOUNDS_M[3], GRID_STEP_M)
    shape = (len(CAMERAS), len(ys), len(xs), 2, 2)
    precision = {name: np.empty(shape, dtype=float) for name in ("M0", "M1", "M2")}
    support = np.full((len(CAMERAS), len(ys), len(xs)), np.inf, dtype=float)
    r1 = {
        camera: model["per_camera_covariance_ray_m2"][index]
        for index, camera in enumerate(CAMERAS)
    }

    for camera_index, camera in enumerate(CAMERAS):
        for y_index, y in enumerate(ys):
            for x_index, x in enumerate(xs):
                point = np.asarray([x, y], dtype=float)
                basis = ray_basis(camera_xy[camera], point)
                ray_covariance = {
                    "M0": model["global_covariance_ray_m2"],
                    "M1": r1[camera],
                }
                ray_covariance["M2"], local_support = spatial_ray_covariance(
                    model, camera, point)
                support[camera_index, y_index, x_index] = local_support
                for name, covariance in ray_covariance.items():
                    world_covariance = basis @ covariance @ basis.T
                    precision[name][camera_index, y_index, x_index] = psd(
                        np.linalg.inv(world_covariance))

    source_hashes = {
        "covariance_models.npz": sha256(model_path),
        "warehouse_v2.world.sdf": sha256(world),
        "planning_precision.py": sha256(Path(__file__)),
    }
    artifacts = {}
    for name, runtime_name in (
        ("M0", "R0_global_full"),
        ("M1", "R1_per_camera_full"),
        ("M2", "R2_spatial_residual"),
    ):
        metadata = {
            "schema": "camera_network.matched_covariance_precision.v1",
            "reference": "robot_ground_reference_xy",
            "frame": "map_bev",
            "planning_target": "inverse_of_matched_runtime_covariance",
            "planner_model": name,
            "runtime_covariance_source": runtime_name,
            "covariance_query": "same_position_and_camera_as_planning_query",
            "additional_planning_fit": False,
            "detector_opportunities_used": False,
            "gate_outcomes_used": False,
            "D_eval_accessed": False,
            "grid_step_m": GRID_STEP_M,
            "grid_bounds_m": GRID_BOUNDS_M,
            "source_hashes": source_hashes,
        }
        path = staging / f"{name.lower()}_planning_precision.npz"
        write_artifact(path, xs, ys, precision[name], support, metadata)
        artifacts[name] = {"path": path.name, "sha256": sha256(path)}

    report = {
        "schema": "matched_covariance_planning_precision_manifest.v1",
        "status": "frozen_before_navigation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_matching": {
            "M0": "R0_global_full",
            "M1": "R1_per_camera_full",
            "M2": "R2_spatial_residual",
        },
        "method": "inverse of matched runtime covariance at each camera and grid position",
        "additional_planning_fit": False,
        "detector_opportunities_used": False,
        "gate_outcomes_used": False,
        "D_eval_accessed": False,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    report_path = staging / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(
        json.dumps({"manifest_sha256": sha256(report_path)}) + "\n", encoding="utf-8")
    os.replace(staging, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
