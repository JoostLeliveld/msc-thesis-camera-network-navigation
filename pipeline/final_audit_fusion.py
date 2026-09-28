#!/usr/bin/env python3
"""Evaluate frozen R0/R1/R2 fusion on sealed final-audit camera batches."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "world"),
                str(REPO), str(REPO / "src/reliability"),
                str(REPO / "src/unav_common")]

from pipeline.dataset import (  # noqa: E402
    image_path, load_rows,
)
from reliability.commissioned_visibility import CommissionedVisibilitySensorModel  # noqa: E402
from reliability.contracts import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world, project_observation_to_world_with_covariance,
)
from unav_common.visibility_patch import visibility_grid_from_bgr_frame  # noqa: E402


MODEL_PATHS = {
    "global": "R0_global_full.json",
    "per_camera": "R1_per_camera_full.json",
    "spatial": "R2_spatial_residual.json",
}
CHI2_95 = 5.9914645471


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fuse(items: list[dict], model: str) -> tuple[np.ndarray, np.ndarray]:
    precision = np.zeros((2, 2), dtype=float)
    weighted = np.zeros(2, dtype=float)
    for item in items:
        covariance = np.asarray(item["covariance_m2"][model], dtype=float)
        information = np.linalg.inv(covariance)
        precision += information
        weighted += information @ np.asarray(item["corrected_xy_m"], dtype=float)
    covariance = np.linalg.inv(precision)
    return covariance @ weighted, covariance


def summary(errors: list[float], nis: list[float] | None = None,
            areas: list[float] | None = None) -> dict:
    values = np.asarray(errors, dtype=float)
    result = {
        "batches": int(len(values)),
        "rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "mean_error_m": float(np.mean(values)),
        "median_error_m": float(np.median(values)),
        "p95_error_m": float(np.quantile(values, 0.95)),
    }
    if nis is not None:
        scores = np.asarray(nis, dtype=float)
        result.update({
            "mean_nis": float(np.mean(scores)),
            "coverage_95": float(np.mean(scores <= CHI2_95)),
        })
    if areas is not None:
        result["mean_fused_ellipse_area_95_cm2"] = float(np.mean(areas))
    return result


def single_observation_summary(items: list[dict], model: str) -> dict:
    by_position: dict[str, list[tuple[float, float, float]]] = collections.defaultdict(list)
    for item in items:
        truth = np.asarray(item["truth_xy_m"], dtype=float)
        estimate = np.asarray(item["corrected_xy_m"], dtype=float)
        covariance = np.asarray(item["covariance_m2"][model], dtype=float)
        residual = estimate - truth
        score = float(residual @ np.linalg.solve(covariance, residual))
        logdet = float(np.linalg.slogdet(covariance)[1])
        nll = 0.5 * (2.0 * math.log(2.0 * math.pi) + logdet + score)
        area = math.pi * CHI2_95 * math.sqrt(math.exp(logdet)) * 1.0e4
        by_position[item["position_key"]].append((nll, score, area))
    return {
        "observations": len(items),
        "positions": len(by_position),
        "equal_position_mean_nll": float(np.mean([
            np.mean([value[0] for value in group]) for group in by_position.values()])),
        "equal_position_mean_nis": float(np.mean([
            np.mean([value[1] for value in group]) for group in by_position.values()])),
        "equal_position_coverage_95": float(np.mean([
            np.mean([value[1] <= CHI2_95 for value in group])
            for group in by_position.values()])),
        "equal_position_mean_ellipse_area_95_cm2": float(np.mean([
            np.mean([value[2] for value in group]) for group in by_position.values()])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rproj-ddev", type=Path, default=None,
        help="D_dev evaluation manifest holding the Rproj sigma_px fitted on D_R; adds the "
             "geometric baseline as a fusion rule")
    args = parser.parse_args()
    audit = args.audit.resolve()
    runtime_root = args.runtime_root.resolve()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)

    audit_report_path = audit / "report.json"
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    records_path = audit / audit_report["records"]["path"]
    if (audit_report.get("status") != "complete"
            or not audit_report.get("final_audit_accessed")
            or audit_report.get("selection_or_fitting_performed")
            or sha256(records_path) != audit_report["records"]["sha256"]):
        raise RuntimeError("sealed final-audit evidence is not valid")

    models = {
        name: CommissionedVisibilitySensorModel(runtime_root / filename)
        for name, filename in MODEL_PATHS.items()
    }
    runtime_hashes = {
        name: sha256(runtime_root / filename)
        for name, filename in MODEL_PATHS.items()
    }
    by_hash = {
        row["image_sha1"]: row for row in load_rows()
        if row["stratum"] == "final_audit"
    }
    source = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    admitted = [row for row in source if row["outcome"] == "admitted"]
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    cameras = {}
    for row in by_hash.values():
        if row["camera_id"] not in cameras:
            cameras[row["camera_id"]] = camera_model_from_world(
                world, include_name=row["camera_model"])
    rproj_sigma_px = None
    fused_rules = tuple(MODEL_PATHS)
    if args.rproj_ddev is not None:
        rproj_sigma_px = float(json.loads(args.rproj_ddev.read_text(encoding="utf-8"))[
            "Rproj"]["sigma_px"])
        fused_rules = fused_rules + ("rproj",)

    evaluated = []
    for index, record in enumerate(admitted, start=1):
        capture = by_hash[record["image_sha1"]]
        box = np.asarray(record["bbox_xyxy"], dtype=float)
        raw = np.asarray(record["raw_xy_m"], dtype=float)
        truth = np.asarray(record["truth_xy_m"], dtype=float)
        frame = cv2.imread(str(image_path(capture)), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(image_path(capture))
        grid = visibility_grid_from_bgr_frame(frame, box)
        item = {
            "position_key": record["position_key"],
            "yaw_idx": int(record["yaw_idx"]),
            "camera_id": record["camera_id"],
            "truth_xy_m": truth.tolist(),
            "corrected_xy_m": None,
            "covariance_m2": {},
        }
        for name, model in models.items():
            estimate, covariance = model.correct_and_covariance(
                record["camera_id"], raw, box, float(record["confidence"]),
                grid, float(capture["robot_yaw"]))
            estimate = np.asarray(estimate, dtype=float)
            if item["corrected_xy_m"] is None:
                item["corrected_xy_m"] = estimate.tolist()
            elif not np.allclose(estimate, item["corrected_xy_m"], rtol=0.0, atol=1e-12):
                raise RuntimeError("mean correction differs between covariance models")
            item["covariance_m2"][name] = np.asarray(covariance, dtype=float).tolist()
        if rproj_sigma_px is not None:
            projected = project_observation_to_world_with_covariance(CameraObservation(
                camera_id=record["camera_id"], timestamp_s=0.0,
                pixel_uv=(0.5 * (box[0] + box[2]), box[3]),
                detection_valid=True, detector_score=1.0,
                conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)),
            ), cameras[record["camera_id"]])
            if projected is None:
                raise RuntimeError("Rproj projection failed at an admitted observation")
            item["covariance_m2"]["rproj"] = (
                rproj_sigma_px ** 2 * np.asarray(projected[1], dtype=float)).tolist()
        evaluated.append(item)
        if index % 100 == 0:
            print(f"final-audit fusion preparation {index}/{len(admitted)}", flush=True)

    batches: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for item in evaluated:
        batches[(item["position_key"], item["yaw_idx"])].append(item)
    batches = {key: value for key, value in batches.items() if len(value) >= 2}
    if not batches:
        raise RuntimeError("final audit has no multi-camera batches")

    errors: dict[str, list[float]] = collections.defaultdict(list)
    nis: dict[str, list[float]] = collections.defaultdict(list)
    areas: dict[str, list[float]] = collections.defaultdict(list)
    by_count: dict[str, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    spatial_wins = 0
    for items in batches.values():
        truth = np.asarray(items[0]["truth_xy_m"], dtype=float)
        equal = np.mean([item["corrected_xy_m"] for item in items], axis=0)
        errors["equal"].append(float(np.linalg.norm(equal - truth)))
        count = len(items)
        by_count["equal"][count].append(errors["equal"][-1])

        best = min(items, key=lambda item: np.linalg.det(
            np.asarray(item["covariance_m2"]["spatial"], dtype=float)))
        best_error = float(np.linalg.norm(
            np.asarray(best["corrected_xy_m"], dtype=float) - truth))
        errors["best_spatial_single"].append(best_error)
        by_count["best_spatial_single"][count].append(best_error)

        # Naive single-camera rule: the camera mounted nearest to its own corrected
        # estimate. Uses no covariance model and no ground truth.
        closest = min(items, key=lambda item: np.linalg.norm(
            np.asarray(item["corrected_xy_m"], dtype=float)
            - cameras[item["camera_id"]].cam_pos[:2]))
        errors["closest_single"].append(float(np.linalg.norm(
            np.asarray(closest["corrected_xy_m"], dtype=float) - truth)))
        by_count["closest_single"][count].append(errors["closest_single"][-1])

        for name in fused_rules:
            estimate, covariance = fuse(items, name)
            residual = estimate - truth
            value = float(np.linalg.norm(residual))
            errors[name].append(value)
            nis[name].append(float(residual @ np.linalg.solve(covariance, residual)))
            areas[name].append(math.pi * CHI2_95 * math.sqrt(np.linalg.det(covariance)) * 1e4)
            by_count[name][count].append(value)
        spatial_wins += errors["spatial"][-1] < best_error

    metrics = {
        "best_spatial_single": summary(errors["best_spatial_single"]),
        "closest_single": summary(errors["closest_single"]),
        "equal": summary(errors["equal"]),
        **{name: summary(errors[name], nis[name], areas[name]) for name in fused_rules},
    }
    metrics["spatial"]["wins_against_best_spatial_single"] = spatial_wins
    metrics["spatial"]["win_fraction_against_best_spatial_single"] = (
        spatial_wins / len(batches))
    metrics["spatial"]["rmse_reduction_vs_equal"] = (
        1.0 - metrics["spatial"]["rmse_m"] / metrics["equal"]["rmse_m"])
    metrics["spatial"]["rmse_reduction_vs_closest_single"] = (
        1.0 - metrics["spatial"]["rmse_m"] / metrics["closest_single"]["rmse_m"])
    metrics["spatial"]["rmse_reduction_vs_best_spatial_single"] = (
        1.0 - metrics["spatial"]["rmse_m"]
        / metrics["best_spatial_single"]["rmse_m"])
    count_metrics = {
        method: {
            str(count): summary(values)
            for count, values in sorted(groups.items())
        }
        for method, groups in by_count.items()
    }

    report = {
        "schema": "thesis_final_audit_fusion.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_or_fitting_performed": False,
        "sample_unit": "position-heading camera batch",
        "inclusion": "at least two admitted cameras in the sealed final audit",
        "fusion_rule": "independent information-form fusion",
        "best_single_rule": "admitted camera with smallest spatial-model covariance determinant",
        "closest_single_rule": "admitted camera mounted nearest (ground plane) to its own corrected estimate",
        "batches": len(batches),
        "admitted_observations": len(admitted),
        "metrics": metrics,
        "by_camera_count": count_metrics,
        "single_observation": {
            name: single_observation_summary(evaluated, name)
            for name in fused_rules
        },
        "rproj_sigma_px": rproj_sigma_px,
        "source_hashes": {
            "audit_report": sha256(audit_report_path),
            "audit_records": sha256(records_path),
            **{f"runtime_{name}": value for name, value in runtime_hashes.items()},
            "implementation": sha256(Path(__file__)),
        },
    }
    staging.mkdir(parents=True)
    records_output = staging / "evaluated_admitted.jsonl"
    with records_output.open("w", encoding="utf-8") as handle:
        for item in evaluated:
            handle.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
    report["records"] = {
        "path": records_output.name,
        "sha256": sha256(records_output),
    }
    report_path = staging / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    (staging / ".complete").write_text(
        json.dumps({"report_sha256": sha256(report_path)}) + "\n",
        encoding="utf-8")
    os.replace(staging, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
