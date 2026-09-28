#!/usr/bin/env python3
"""Evaluate one detector candidate on the frozen, real spatial validation split.

This evaluator deliberately does not read commissioning_fit or final_audit data.  It
uses the Stage-04 label ledger as its complete population and reports both ranked AP
and fixed-threshold operating characteristics, including robot-absent false alarms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[2]
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
CONFIDENCE_THRESHOLDS = (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=float)
    x0 = np.maximum(boxes[:, 0], gt[0])
    y0 = np.maximum(boxes[:, 1], gt[1])
    x1 = np.minimum(boxes[:, 2], gt[2])
    y1 = np.minimum(boxes[:, 3], gt[3])
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    area_p = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    area_g = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    return inter / np.maximum(area_p + area_g - inter, 1e-12)


def average_precision(tp: np.ndarray, scores: np.ndarray, positives: int) -> float:
    if positives == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / positives
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, 101)
    return float(np.mean([np.max(precision[recall >= x]) for x in grid]))


def width_bin(width: float) -> str:
    if width < 32:
        return "16-31"
    if width < 64:
        return "32-63"
    return ">=64"


def operating_point(records: list[dict], threshold: float) -> dict:
    tp = fp = fn = 0
    negative_detections = 0
    negative_images = 0
    group_totals: dict[str, dict[str, list[int]]] = {
        "camera": defaultdict(lambda: [0, 0]),
        "visible_width_px": defaultdict(lambda: [0, 0]),
    }
    for row in records:
        keep = row["scores"] >= threshold
        overlaps = row["ious"][keep]
        detections = int(np.count_nonzero(keep))
        if row["positive"]:
            matched = bool(np.any(overlaps >= 0.5))
            tp += int(matched)
            fn += int(not matched)
            fp += detections - int(matched)
            group_totals["camera"][row["camera"]][1] += 1
            group_totals["camera"][row["camera"]][0] += int(matched)
            wb = width_bin(row["width"])
            group_totals["visible_width_px"][wb][1] += 1
            group_totals["visible_width_px"][wb][0] += int(matched)
        else:
            fp += detections
            negative_detections += detections
            negative_images += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "confidence": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "negative_images": negative_images,
        "false_detections_on_negatives": negative_detections,
        "false_detections_per_negative_image": negative_detections / max(negative_images, 1),
        "recall_by_camera": {
            key: {"hits": value[0], "positives": value[1], "recall": value[0] / value[1]}
            for key, value in sorted(group_totals["camera"].items())
        },
        "recall_by_visible_width_px": {
            key: {"hits": value[0], "positives": value[1], "recall": value[0] / value[1]}
            for key, value in sorted(group_totals["visible_width_px"].items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    trial_path = args.trial.resolve()
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    if trial.get("status") != "complete_pending_cross_resolution_evaluation":
        raise RuntimeError("Trial is not complete and ready for evaluation")
    if trial["protocol_sha256"] != sha256(protocol_path):
        raise RuntimeError("Training protocol hash mismatch")
    if trial["dataset_yaml_sha256"] != protocol["data"]["yaml_sha256"]:
        raise RuntimeError("Dataset hash mismatch")

    dataset_root = (REPO / protocol["data"]["yaml"]).resolve().parent
    ledger_path = dataset_root / "label_ledger.csv"
    rows = list(csv.DictReader(ledger_path.open(encoding="utf-8", newline="")))
    if any(row["dataset_split"] not in {"detector_fit", "detector_validation"} for row in rows):
        raise RuntimeError("Stage-04 ledger contains an unauthorized dataset role")
    val = [row for row in rows if row["dataset_split"] == "detector_validation" and row["label_class"] != "ambiguous"]
    if len(val) != 1404:
        raise RuntimeError(f"Expected 1404 eligible validation images, found {len(val)}")
    images = [(dataset_root / row["exported_image"]).resolve() for row in val]
    if any(not path.is_file() for path in images):
        raise RuntimeError("A frozen validation image is missing")

    weights = Path(trial["candidate_weights"]).resolve()
    if sha256(weights) != trial["candidate_weights_sha256"]:
        raise RuntimeError("Candidate checkpoint hash mismatch")
    model = YOLO(str(weights))
    # Ultralytics treats a Python list of paths as one in-memory source batch, even
    # with ``stream=True``.  Chunk explicitly so the frozen trial batch size is also
    # an actual upper bound on evaluator GPU memory.
    def predict_in_bounded_batches():
        batch = int(trial["batch"])
        for start in range(0, len(images), batch):
            chunk = images[start : start + batch]
            yield from model.predict(
                source=[str(path) for path in chunk], imgsz=int(trial["imgsz"]), conf=0.001,
                iou=0.7, max_det=20, device=str(protocol["common"]["device"]),
                batch=batch, workers=int(protocol["common"]["workers"]),
                verbose=False, save=False, stream=False,
            )

    results = predict_in_bounded_batches()

    records: list[dict] = []
    prediction_rows: list[dict] = []
    for row, result in zip(val, results, strict=True):
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
        scores = result.boxes.conf.detach().cpu().numpy().astype(float)
        positive = row["label_class"] == "positive"
        if positive:
            gt = np.array([float(row[k]) for k in ("mask_x0", "mask_y0", "mask_x1", "mask_y1")])
            overlaps = iou(boxes, gt)
            width = float(row["bbox_width_px"])
        else:
            overlaps = np.zeros(len(scores), dtype=float)
            width = float("nan")
        records.append({
            "positive": positive, "camera": row["camera_id"], "width": width,
            "scores": scores, "ious": overlaps,
        })
        prediction_rows.append({
            "pose_id": int(row["pose_id"]), "position_id": int(row["position_id"]),
            "camera_id": row["camera_id"], "label_class": row["label_class"],
            "exported_image": row["exported_image"],
            "predictions": [
                {"xyxy": [float(x) for x in box], "confidence": float(score), "iou_gt": float(overlap)}
                for box, score, overlap in zip(boxes, scores, overlaps, strict=True)
            ],
        })

    positives = sum(row["positive"] for row in records)
    aps = []
    for threshold in IOU_THRESHOLDS:
        scores_all, tp_all = [], []
        for row in records:
            matched = False
            order = np.argsort(-row["scores"], kind="stable")
            for index in order:
                hit = bool(row["positive"] and not matched and row["ious"][index] >= threshold)
                matched |= hit
                scores_all.append(row["scores"][index])
                tp_all.append(hit)
        aps.append(average_precision(np.asarray(tp_all), np.asarray(scores_all), positives))

    operating_points = [operating_point(records, threshold) for threshold in CONFIDENCE_THRESHOLDS]
    report = {
        "schema": "thesis_detector_validation_report.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "population": "detector_validation only; 100% real; ambiguous excluded",
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "trial_manifest": str(trial_path), "trial_manifest_sha256": sha256(trial_path),
        "checkpoint": str(weights), "checkpoint_sha256": sha256(weights),
        "imgsz": int(trial["imgsz"]), "images": len(records), "positives": positives,
        "negatives": len(records) - positives, "iou_matching_threshold": 0.5,
        "ap50": aps[0], "ap50_95": float(np.mean(aps)),
        "ap_by_iou": {f"{x:.2f}": y for x, y in zip(IOU_THRESHOLDS, aps, strict=True)},
        "operating_points": operating_points,
        "primary_fixed_operating_point": next(x for x in operating_points if x["confidence"] == 0.25),
        "selection_warning": "Do not select from aggregate AP alone; apply the frozen cross-resolution rule.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    raw_path = args.output.with_name(args.output.stem + "_predictions.jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({k: report[k] for k in ("imgsz", "images", "positives", "negatives", "ap50", "ap50_95")}, indent=2))
    print(f"fixed@0.25={json.dumps(report['primary_fixed_operating_point'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
