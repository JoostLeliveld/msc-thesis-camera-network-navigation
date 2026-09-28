#!/usr/bin/env python3
"""Evaluate the sealed final-audit partition once, without fitting: correction, R0/R1/R2 and R_proj."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/perception"),
                str(REPO / "src/planning"), str(REPO / "src/reliability"),
                str(REPO / "src/unav_common")]

from pipeline.detector.export_dataset import classify  # noqa: E402
from pipeline.dataset import (  # noqa: E402
    image_path, load_rows,
)
from perception.core.yolo_selection import select_best_detection, target_class_ids  # noqa: E402
from planning.core.camera_network import CameraNetworkModel  # noqa: E402
from reliability.commissioned_visibility import CommissionedVisibilitySensorModel  # noqa: E402
from reliability.observation_gates import (  # noqa: E402
    UsableObservationGateConfig, evaluate_sensor_gate,
)
from reliability.contracts import CameraObservation  # noqa: E402
from reliability.projection import (  # noqa: E402
    camera_model_from_world, project_observation_to_world_with_covariance,
)
from unav_common.capture_integrity import checked_image  # noqa: E402
from unav_common.visibility_patch import visibility_grid_from_bgr_frame  # noqa: E402

CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
MODEL_KEYS = ("R0_global_full", "R1_per_camera_full", "R2_spatial_full")
# protocol locked-input keys of each model's runtime package and planning field
RUNTIME_INPUT = {"R0_global_full": "runtime_R0", "R1_per_camera_full": "runtime_R1",
                 "R2_spatial_full": "runtime_R2"}
PLANNING_INPUT = {"R0_global_full": "planning_M0", "R1_per_camera_full": "planning_M1",
                  "R2_spatial_full": "planning_M2"}
CHI2 = {"50": 1.3862943611, "90": 4.605170186,
        "95": 5.9914645471, "99": 9.210340372}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def position_summary(rows: list[dict], value: str) -> dict:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        if row.get(value) is not None:
            grouped[row["position_key"]].append(float(row[value]))
    means = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
    pooled = np.asarray([float(row[value]) for row in rows if row.get(value) is not None])
    if not len(pooled):
        return {"observations": 0, "positions": 0}
    return {
        "observations": int(len(pooled)), "positions": int(len(means)),
        "pooled_median": float(np.median(pooled)),
        "pooled_p95": float(np.quantile(pooled, 0.95)),
        "pooled_rmse": float(np.sqrt(np.mean(pooled ** 2))),
        "equal_position_mean": float(np.mean(means)),
        "equal_position_rmse": float(np.sqrt(np.mean(means ** 2))),
    }


def covariance_summary(rows: list[dict], model: str) -> dict:
    valid = [row for row in rows if model in row.get("mahalanobis_d2", {})]
    by_position: dict[str, list[dict]] = collections.defaultdict(list)
    for row in valid:
        by_position[row["position_key"]].append(row)
    def nll(row: dict) -> float:
        return 0.5 * (2.0 * math.log(2.0 * math.pi)
                      + row["logdet_covariance"][model]
                      + row["mahalanobis_d2"][model])
    return {
        "observations": len(valid), "positions": len(by_position),
        "equal_position_mean_nll": float(np.mean([
            np.mean([nll(row) for row in group]) for group in by_position.values()])),
        "equal_position_mean_nis": float(np.mean([
            np.mean([row["mahalanobis_d2"][model] for row in group])
            for group in by_position.values()])),
        "equal_position_coverage": {
            level: float(np.mean([
                np.mean([row["mahalanobis_d2"][model] <= threshold for row in group])
                for group in by_position.values()]))
            for level, threshold in CHI2.items()
        },
        "equal_position_mean_ellipse_area_95_cm2": float(np.mean([
            np.mean([math.pi * CHI2["95"] * math.sqrt(
                math.exp(row["logdet_covariance"][model])) * 1e4 for row in group])
            for group in by_position.values()])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    protocol_path, output = args.protocol.resolve(), args.output.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_final_audit_access":
        raise RuntimeError("final-audit protocol is not frozen")
    if output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError(output)
    for entry in protocol["locked_inputs"].values():
        path = REPO / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"locked input drift: {path}")

    rows = [row for row in load_rows() if row["stratum"] == "final_audit"]
    expected = protocol["population"]
    if (len(rows) != int(expected["opportunities"])
            or len({row["position_key"] for row in rows}) != int(expected["positions"])):
        raise RuntimeError("final-audit population differs from the campaign lock")
    if any(row["capture_status"] != "ok" for row in rows):
        raise RuntimeError("final-audit population contains a failed capture")

    lock = json.loads((REPO / protocol["locked_inputs"]["campaign_lock"]["path"])
                      .read_text(encoding="utf-8"))
    weights = REPO / lock["detector"]["checkpoint"]
    gate = UsableObservationGateConfig.from_yaml(str(
        REPO / protocol["locked_inputs"]["gate_config"]["path"]))
    gate.assert_belief_independent()
    # The exact files the protocol hashed: the runtime packages and the matched-covariance
    # planning fields (the planning artifact the canonical lock prescribes).
    locked = protocol["locked_inputs"]
    runtime = {
        key: CommissionedVisibilitySensorModel(REPO / locked[RUNTIME_INPUT[key]]["path"])
        for key in MODEL_KEYS
    }
    planning = {
        key: CameraNetworkModel(REPO / locked[PLANNING_INPUT[key]]["path"])
        for key in MODEL_KEYS
    }
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    cameras = {camera: camera_model_from_world(world, include_name=next(
        row["camera_model"] for row in rows if row["camera_id"] == camera))
        for camera in CAMERAS}
    # R_proj, the external baseline: sigma_px^2 I2 propagated through the pixel-to-ground
    # Jacobian at the raw box bottom centre, scored on the same corrected observations.
    # sigma_px is the value fitted on D_R by evaluate_ddev.py. It is optional so
    # the v5 protocol, which predates it, still runs unchanged.
    rproj_sigma_px = None
    if "ddev_evaluation" in protocol["locked_inputs"]:
        ddev = json.loads((REPO / protocol["locked_inputs"]["ddev_evaluation"]["path"])
                          .read_text(encoding="utf-8"))
        rproj_sigma_px = float(ddev["Rproj"]["sigma_px"])
    label_path = REPO / protocol["locked_inputs"]["label_protocol"]["path"]
    label_contract = json.loads(label_path.read_text(encoding="utf-8"))[
        "detector_label_contract"]["positive_requires_all"]

    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["image_sha1"], row)
    selections = {}
    with tempfile.TemporaryDirectory(prefix="final_audit_detector_") as directory:
        private_weights = Path(directory) / weights.name
        private_weights.write_bytes(weights.read_bytes())
        detector = YOLO(str(private_weights))
        ids = target_class_ids(getattr(detector, "names", {}), "robot", -1)
        if not ids and set(getattr(detector, "names", {})) == {0}:
            ids = {0}
        items = sorted(unique.items())
        started = time.monotonic()
        for start in range(0, len(items), args.batch_size):
            chunk = items[start:start + args.batch_size]
            images = [checked_image(
                image_path(row).parent.parent.parent, row,
                size=(1280, 720)) for _, row in chunk]
            results = detector.predict(
                source=images, imgsz=int(lock["detector"]["imgsz"]), conf=0.001,
                iou=0.45, batch=len(images), device=str(args.device),
                stream=False, verbose=False)
            for (image_hash, _), result in zip(chunk, results, strict=True):
                selections[image_hash] = select_best_detection(
                    result, target_ids=ids, confidence_threshold=0.25,
                    use_masks=False, mask_min_area=0.0, mask_bottom_band_px=3.0)
            print(f"final audit detector {min(start + len(chunk), len(items))}/{len(items)}",
                  flush=True)
        inference_elapsed = time.monotonic() - started

    evaluated = []
    label_counts = collections.Counter()
    outcome_counts = collections.Counter()
    for index, row in enumerate(rows, start=1):
        selection = selections[row["image_sha1"]]
        box = selection["bbox_xyxy"]
        detected = bool(selection["detected"])
        raw = None
        if detected and box is not None:
            raw = cameras[row["camera_id"]].pixel_to_world_at_z(
                0.5 * (float(box[0]) + float(box[2])), float(box[3]), 0.0)
        gate_result = evaluate_sensor_gate({
            "detection_received": detected,
            "detector_confidence": float(selection["confidence"]),
            "bbox_xmin": None if box is None else box[0],
            "bbox_ymin": None if box is None else box[1],
            "bbox_xmax": None if box is None else box[2],
            "bbox_ymax": None if box is None else box[3],
            "projection_valid": raw is not None,
        }, gate)
        outcome = "admitted" if gate_result.admitted else (
            "detector_miss" if gate_result.reason == "NO_DETECTION" else "gate_refusal")
        label, reasons, mask_box, _ = classify(row, 1280, 720, label_contract)
        label_counts[label] += 1; outcome_counts[outcome] += 1
        truth = np.asarray([float(row["robot_x"]), float(row["robot_y"])])
        result = {
            "position_key": row["position_key"], "yaw_idx": int(row["yaw_idx"]),
            "camera_id": row["camera_id"], "image_sha1": row["image_sha1"],
            "reference_class": label, "reference_reasons": reasons,
            "mask_box_xyxy": None if mask_box is None else list(mask_box),
            "detected": detected, "confidence": float(selection["confidence"]),
            "bbox_xyxy": None if box is None else [float(value) for value in box],
            "gate_reason": gate_result.reason, "outcome": outcome,
            "truth_xy_m": truth.tolist(), "raw_xy_m": None if raw is None else list(raw[:2]),
            "raw_error_m": None if raw is None else float(np.linalg.norm(np.asarray(raw[:2]) - truth)),
            "corrected_error_m": None, "mahalanobis_d2": {},
            "logdet_covariance": {}, "planning_error_fro": {},
        }
        corrected = None
        if gate_result.admitted:
            image = cv2.imread(str(image_path(row)), cv2.IMREAD_COLOR)
            grid = visibility_grid_from_bgr_frame(image, box)
            for key in MODEL_KEYS:
                estimate, covariance = runtime[key].correct_and_covariance(
                    row["camera_id"], raw[:2], box, float(selection["confidence"]),
                    grid, float(row["robot_yaw"]))
                residual = np.asarray(estimate) - truth
                matrix = np.asarray(covariance)
                result["mahalanobis_d2"][key] = float(
                    residual @ np.linalg.solve(matrix, residual))
                result["logdet_covariance"][key] = float(np.linalg.slogdet(matrix)[1])
                realized = np.linalg.inv(matrix)
                predicted = planning[key].query(
                    [truth[0], truth[1], float(row["robot_yaw"])])[
                        "expected_information"][CAMERAS.index(row["camera_id"])]
                result["planning_error_fro"][key] = float(np.linalg.norm(predicted - realized))
                if corrected is None:
                    corrected = np.asarray(estimate)
            result["corrected_error_m"] = float(np.linalg.norm(corrected - truth))
            if rproj_sigma_px is not None:
                bottom = (0.5 * (float(box[0]) + float(box[2])), float(box[3]))
                projected = project_observation_to_world_with_covariance(CameraObservation(
                    camera_id=row["camera_id"], timestamp_s=0.0, pixel_uv=bottom,
                    detection_valid=True, detector_score=1.0,
                    conditional_cov_uv=((1.0, 0.0), (0.0, 1.0)),
                ), cameras[row["camera_id"]])
                if projected is None:
                    raise RuntimeError("Rproj projection failed at an admitted observation")
                matrix = rproj_sigma_px ** 2 * np.asarray(projected[1], dtype=float)
                residual = corrected - truth
                result["mahalanobis_d2"]["Rproj"] = float(
                    residual @ np.linalg.solve(matrix, residual))
                result["logdet_covariance"]["Rproj"] = float(np.linalg.slogdet(matrix)[1])
        else:
            for key in MODEL_KEYS:
                predicted = planning[key].query(
                    [truth[0], truth[1], float(row["robot_yaw"])])[
                        "expected_information"][CAMERAS.index(row["camera_id"])]
                result["planning_error_fro"][key] = float(np.linalg.norm(predicted))
        evaluated.append(result)
        if index % 500 == 0:
            print(f"final audit evaluation {index}/{len(rows)}", flush=True)

    eligible = [row for row in evaluated if row["reference_class"] in ("positive", "negative")]
    tp = sum(row["reference_class"] == "positive" and row["detected"] for row in eligible)
    fp = sum(row["reference_class"] == "negative" and row["detected"] for row in eligible)
    fn = sum(row["reference_class"] == "positive" and not row["detected"] for row in eligible)
    admitted = [row for row in evaluated if row["outcome"] == "admitted"]
    report = {
        "schema": "thesis_reference_final_audit.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_audit_accessed": True, "selection_or_fitting_performed": False,
        "sample_unit": "complete physical position",
        "population": {"positions": int(expected["positions"]),
                       "opportunities": int(expected["opportunities"]),
                       "unique_images": len(unique)},
        "reference_class_counts": dict(sorted(label_counts.items())),
        "opportunity_outcomes": dict(sorted(outcome_counts.items())),
        "detector": {"true_positive": tp, "false_positive": fp, "false_negative": fn,
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)},
        "admission": {"admitted": len(admitted),
                      "precision_positive": sum(r["reference_class"] == "positive" for r in admitted)
                      / max(len(admitted), 1)},
        "raw_error": position_summary(admitted, "raw_error_m"),
        "corrected_error": position_summary(admitted, "corrected_error_m"),
        "covariance": {key: covariance_summary(admitted, key) for key in MODEL_KEYS
                       + (("Rproj",) if rproj_sigma_px is not None else ())},
        "rproj_sigma_px": rproj_sigma_px,
        "planning_information": {
            key: position_summary(evaluated, f"planning_error_fro.{key}") for key in ()
        },
        "inference_elapsed_s": inference_elapsed,
        "protocol": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": sha256(protocol_path),
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    report["planning_information"] = {}
    for key in MODEL_KEYS:
        values = [{**row, "value": row["planning_error_fro"][key]} for row in evaluated]
        report["planning_information"][key] = position_summary(values, "value")

    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir(parents=True)
    records_path = staging / "evaluated.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    report["records"] = {"path": records_path.name, "sha256": sha256(records_path)}
    report_path = staging / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(
        json.dumps({"report_sha256": sha256(report_path)}) + "\n", encoding="utf-8")
    os.replace(staging, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
