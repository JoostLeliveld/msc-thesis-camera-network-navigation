#!/usr/bin/env python3
"""Apply the frozen runtime gate and build canonical correction-fit tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/reliability"), str(REPO / "src/unav_common")]

from pipeline.detector.export_dataset import classify  # noqa: E402
from pipeline.dataset import (  # noqa: E402
    EXPECTED_WORKING_OPPORTUNITIES, image_path, load_rows,
)
from reliability.observation_gates import UsableObservationGateConfig, evaluate_sensor_gate  # noqa: E402
from reliability.projection import camera_model_from_world  # noqa: E402
from unav_common.visibility_patch import visibility_grid_from_bgr_frame  # noqa: E402

ROLES = ("D_mu", "D_R", "D_dev")
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
FEATURE_NAMES = (
    "raw_range_m", "inverse_raw_range", "ray_bearing_sin", "ray_bearing_cos",
    "bbox_width_fraction", "bbox_height_fraction", "bbox_aspect",
    "bbox_bottom_u_fraction", "bbox_bottom_v_fraction", "confidence",
    *(f"is_camera_{letter}" for letter in "ABCDE"),
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def ray_basis(camera_xy: np.ndarray, point: np.ndarray) -> np.ndarray:
    along = point - camera_xy
    norm = float(np.linalg.norm(along))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("degenerate camera ray")
    along /= norm
    return np.column_stack((along, np.asarray([-along[1], along[0]])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inference, gate_path, output = args.inference.resolve(), args.gate.resolve(), args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)

    manifest_path = inference / "manifest.json"
    records_path = inference / "records.jsonl"
    detector_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if detector_manifest.get("status") != "complete" or detector_manifest.get("final_audit_accessed"):
        raise RuntimeError("detector artifact is incomplete or opened final_audit")
    if detector_manifest.get("records_sha256") != sha256(records_path):
        raise RuntimeError("detector records hash drift")
    gate = UsableObservationGateConfig.from_yaml(str(gate_path))
    gate.assert_belief_independent()

    source_rows = load_rows()
    row_by_key = {
        (r["capture_source"], int(r["plan_pose_index"]), r["camera_id"]): r
        for r in source_rows if r["stratum"] in ROLES
    }
    if len(row_by_key) != EXPECTED_WORKING_OPPORTUNITIES:
        raise RuntimeError(
            f"canonical working population is not {EXPECTED_WORKING_OPPORTUNITIES} opportunities")
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    models = {
        camera: camera_model_from_world(
            world, include_name=next(r["camera_model"] for r in source_rows if r["camera_id"] == camera)
        ) for camera in CAMERAS
    }
    camera_xy = {camera: np.asarray(models[camera].cam_pos[:2], dtype=float) for camera in CAMERAS}
    label_contract_path = REPO / "pipeline/detector/label_protocol.json"
    label_contract = json.loads(label_contract_path.read_text(encoding="utf-8"))[
        "detector_label_contract"
    ]["positive_requires_all"]

    opportunity_counts, reason_counts, dev_confusion = Counter(), Counter(), Counter()
    image_cache: dict[str, np.ndarray] = {}
    admitted: list[dict] = []
    opportunity_path = staging / "opportunities.jsonl"
    with records_path.open(encoding="utf-8") as source, opportunity_path.open("w", encoding="utf-8") as sink:
        for opportunity_index, line in enumerate(source, start=1):
            record = json.loads(line)
            key = (record["capture_source"], int(record["plan_pose_index"]), record["camera_id"])
            row = row_by_key.pop(key)
            box = record.get("bbox_xyxy")
            raw = None
            if box is not None:
                raw = models[record["camera_id"]].pixel_to_world_at_z(
                    0.5 * (float(box[0]) + float(box[2])), float(box[3]), 0.0
                )
            gate_input = {
                "detection_received": bool(record["detected"]),
                "detector_confidence": record["confidence"],
                "bbox_xmin": None if box is None else box[0],
                "bbox_ymin": None if box is None else box[1],
                "bbox_xmax": None if box is None else box[2],
                "bbox_ymax": None if box is None else box[3],
                "projection_valid": raw is not None,
            }
            result = evaluate_sensor_gate(gate_input, gate)
            outcome = "admitted" if result.admitted else (
                "detector_miss" if result.reason == "NO_DETECTION" else "gate_refusal"
            )
            opportunity_counts[(record["stratum"], outcome)] += 1
            reason_counts[result.reason] += 1
            label = None
            if record["stratum"] == "D_dev":
                label = classify(row, 1280, 720, label_contract)[0]
                dev_confusion[(label, outcome)] += 1
            out = {
                **record,
                "gate_id": result.gate_id,
                "gate_config_hash": result.gate_config_hash,
                "gate_reason": result.reason,
                "opportunity_outcome": outcome,
                "raw_ground_xy": None if raw is None else [float(raw[0]), float(raw[1])],
                "offline_reference_class": label,
            }
            sink.write(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n")
            if opportunity_index % 2500 == 0:
                print(
                    f"gate/features {opportunity_index}/{EXPECTED_WORKING_OPPORTUNITIES} opportunities; "
                    f"{len(admitted)} admitted",
                    flush=True,
                )
            if not result.admitted:
                continue
            raw_xy = np.asarray(raw[:2], dtype=float)
            truth = np.asarray([float(record["robot_x"]), float(record["robot_y"])])
            basis = ray_basis(camera_xy[record["camera_id"]], raw_xy)
            dx, dy = raw_xy - camera_xy[record["camera_id"]]
            distance = math.hypot(float(dx), float(dy))
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            feature = np.asarray([
                distance, 1.0 / distance, math.sin(math.atan2(dy, dx)), math.cos(math.atan2(dy, dx)),
                width / 1280.0, height / 720.0, width / height,
                0.5 * (float(box[0]) + float(box[2])) / 1280.0, float(box[3]) / 720.0,
                float(record["confidence"]),
                *[float(record["camera_id"] == camera) for camera in CAMERAS],
            ], dtype=np.float32)
            image_hash = record["image_sha1"]
            if image_hash not in image_cache:
                image = cv2.imread(str(image_path(row)), cv2.IMREAD_COLOR)
                if image is None or image.shape != (720, 1280, 3):
                    raise RuntimeError(f"invalid admitted RGB image: {image_path(row)}")
                image_cache[image_hash] = visibility_grid_from_bgr_frame(image, box)
            admitted.append({
                "role": record["stratum"], "position_key": record["position_key"],
                "plan_pose_index": int(record["plan_pose_index"]), "yaw_idx": int(record["yaw_idx"]),
                "camera": record["camera_id"], "image_sha1": image_hash,
                "truth": truth, "raw": raw_xy, "basis": basis,
                "target_ray": basis.T @ (truth - raw_xy), "feature": feature,
                "grid": image_cache[image_hash], "bbox": np.asarray(box, dtype=np.float32),
                "confidence": float(record["confidence"]),
            })
    if row_by_key:
        raise RuntimeError(f"detector artifact omitted {len(row_by_key)} working opportunities")
    if not admitted:
        raise RuntimeError("gate admitted no observations")

    tensor_path = staging / "admitted.npz"
    np.savez_compressed(
        tensor_path,
        role=np.asarray([x["role"] for x in admitted]),
        position_key=np.asarray([x["position_key"] for x in admitted]),
        plan_pose_index=np.asarray([x["plan_pose_index"] for x in admitted]),
        yaw_idx=np.asarray([x["yaw_idx"] for x in admitted]),
        camera=np.asarray([x["camera"] for x in admitted]),
        image_sha1=np.asarray([x["image_sha1"] for x in admitted]),
        truth_xy_m=np.stack([x["truth"] for x in admitted]),
        raw_xy_m=np.stack([x["raw"] for x in admitted]),
        ray_basis=np.stack([x["basis"] for x in admitted]),
        target_ray_m=np.stack([x["target_ray"] for x in admitted]),
        structured_feature=np.stack([x["feature"] for x in admitted]),
        visibility_grid=np.stack([x["grid"] for x in admitted]),
        bbox_xyxy=np.stack([x["bbox"] for x in admitted]),
        confidence=np.asarray([x["confidence"] for x in admitted]),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    report = {
        "schema": "thesis_reference_gate_dataset.v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_roles": list(ROLES), "final_audit_accessed": False,
        "gate": str(gate_path.relative_to(REPO)), "gate_sha256": sha256(gate_path),
        "gate_config_hash": gate.config_hash(),
        "detector_manifest": str(manifest_path.relative_to(REPO)),
        "detector_manifest_sha256": sha256(manifest_path),
        "label_protocol_sha256": sha256(label_contract_path),
        "opportunity_counts": {
            role: {outcome: opportunity_counts[(role, outcome)] for outcome in (
                "detector_miss", "gate_refusal", "admitted"
            )} for role in ROLES
        },
        "gate_reason_counts": dict(sorted(reason_counts.items())),
        "D_dev_offline_reference_confusion": {
            label: {outcome: dev_confusion[(label, outcome)] for outcome in (
                "detector_miss", "gate_refusal", "admitted"
            )} for label in ("positive", "ambiguous", "negative")
        },
        "admitted_rows": len(admitted), "unique_admitted_images": len(image_cache),
        "feature_names": list(FEATURE_NAMES),
        "runtime_inputs_only": True,
        "opportunities": opportunity_path.name, "opportunities_sha256": sha256(opportunity_path),
        "admitted_tensors": tensor_path.name, "admitted_tensors_sha256": sha256(tensor_path),
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    report_path = staging / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / ".complete").write_text(json.dumps({"manifest_sha256": sha256(report_path)}) + "\n")
    os.replace(staging, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
