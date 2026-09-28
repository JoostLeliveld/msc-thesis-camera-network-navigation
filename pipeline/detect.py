#!/usr/bin/env python3
"""Run the frozen detector over the unsealed v5 reference-survey roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
for rel in ("src/perception", "src/unav_common"):
    sys.path.insert(0, str((REPO / rel).resolve()))

from perception.core.yolo_selection import select_best_detection, target_class_ids  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from unav_common.capture_integrity import checked_image, digest  # noqa: E402
from pipeline.dataset import (  # noqa: E402
    EXPECTED_WORKING_OPPORTUNITIES,
    EXPECTED_WORKING_UNIQUE_IMAGES,
    LOCK_PATH,
    SOURCES,
    image_path,
    load_rows,
)

WORKING_ROLES = ("D_mu", "D_R", "D_dev")
CLIP_EPSILON_PX = 0.5


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    output = args.output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)

    lock_path = LOCK_PATH
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    weights = (REPO / lock["detector"]["checkpoint"]).resolve()
    if sha256(weights) != lock["detector"]["checkpoint_sha256"]:
        raise RuntimeError("frozen detector checkpoint hash drift")

    all_rows = load_rows()
    rows = [row for row in all_rows if row["stratum"] in WORKING_ROLES]
    if len(rows) != EXPECTED_WORKING_OPPORTUNITIES:
        raise RuntimeError(
            f"expected {EXPECTED_WORKING_OPPORTUNITIES} working opportunities, found {len(rows)}")
    if any(row["stratum"] == "final_audit" for row in rows):
        raise RuntimeError("final_audit entered the working population")

    capture_dirs = {name: directory for name, directory, _ in SOURCES}
    manifests = {}
    sizes = {}
    for name, directory in capture_dirs.items():
        manifest_path = directory / "capture_manifest.json"
        manifests[name] = {
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
            "index_sha256": sha256(directory / "capture_index.csv"),
        }
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        sizes[name] = {
            item["camera_id"]: (int(item["image_width"]), int(item["image_height"]))
            for item in meta["cameras"]
        }

    unique = {}
    for row in rows:
        unique.setdefault(row["image_sha1"], row)
    if len(unique) != EXPECTED_WORKING_UNIQUE_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_WORKING_UNIQUE_IMAGES} unique working images, found {len(unique)}")

    items = sorted(unique.items())
    selections = {}
    weights_bytes = weights.read_bytes()
    start_time = time.monotonic()
    staging.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="reference_detector_") as temp_name:
            private_weights = Path(temp_name) / weights.name
            private_weights.write_bytes(weights_bytes)
            model = YOLO(str(private_weights))
            target_ids = target_class_ids(getattr(model, "names", {}), "robot", -1)
            if not target_ids and set(getattr(model, "names", {})) == {0}:
                target_ids = {0}
            if not target_ids:
                raise RuntimeError("frozen detector has no unambiguous robot class")
            batch_size = max(1, int(args.batch_size))
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                images = []
                for _, row in chunk:
                    source = row["capture_source"]
                    images.append(checked_image(
                        capture_dirs[source], row, size=sizes[source][row["camera_id"]]
                    ))
                results = model.predict(
                    source=images,
                    imgsz=int(lock["detector"]["imgsz"]),
                    conf=0.001,
                    iou=0.45,
                    batch=len(images),
                    device=str(args.device),
                    stream=False,
                    verbose=False,
                )
                for (image_hash, _), result in zip(chunk, results, strict=True):
                    selections[image_hash] = select_best_detection(
                        result,
                        target_ids=target_ids,
                        confidence_threshold=0.25,
                        use_masks=False,
                        mask_min_area=0.0,
                        mask_bottom_band_px=3.0,
                    )
                done = min(start + batch_size, len(items))
                if done % 256 < batch_size or done == len(items):
                    print(f"detector {done}/{len(items)} unique images", flush=True)

        outcome_counts = Counter()
        records_path = staging / "records.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                selection = selections[row["image_sha1"]]
                box = selection["bbox_xyxy"]
                detected = bool(selection["detected"])
                width, height = sizes[row["capture_source"]][row["camera_id"]]
                clipped = bool(detected and box is not None and (
                    float(box[0]) <= CLIP_EPSILON_PX
                    or float(box[1]) <= CLIP_EPSILON_PX
                    or float(box[2]) >= width - CLIP_EPSILON_PX
                    or float(box[3]) >= height - CLIP_EPSILON_PX
                ))
                outcome_counts["detected" if detected else "miss"] += 1
                record = {
                    "capture_source": row["capture_source"],
                    "plan_pose_index": int(row["plan_pose_index"]),
                    "position_key": row["position_key"],
                    "yaw_idx": int(row["yaw_idx"]),
                    "stratum": row["stratum"],
                    "kind": row["kind"],
                    "anchor": row["anchor"],
                    "block_id": row["block_id"],
                    "camera_id": row["camera_id"],
                    "image": row["image"],
                    "image_sha1": row["image_sha1"],
                    "robot_x": float(row["robot_x"]),
                    "robot_y": float(row["robot_y"]),
                    "robot_yaw": float(row["robot_yaw"]),
                    "semantic_robot_pixels": int(float(row["semantic_robot_pixels"])),
                    "inference_status": "detected" if detected else "miss",
                    "detected": detected,
                    "detector_clipped": clipped,
                    "n_candidates": int(selection["n_candidates"]),
                    "confidence": float(selection["confidence"]),
                    "bbox_xyxy": None if box is None else [float(value) for value in box],
                    "bbox_bottom_uv": None if not detected else [
                        float(selection["bbox_bottom_u"]), float(selection["bbox_bottom_v"])
                    ],
                }
                handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

        elapsed = time.monotonic() - start_time
        manifest = {
            "schema": "thesis_reference_detector_inference.v1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "authorized_roles": list(WORKING_ROLES),
            "final_audit_accessed": False,
            "opportunity_rows": len(rows),
            "unique_images": len(items),
            "opportunities_by_role": dict(sorted(Counter(r["stratum"] for r in rows).items())),
            "outcomes": dict(sorted(outcome_counts.items())),
            "deduplication_key": "image_sha1",
            "capture_manifests": manifests,
            "campaign_lock": str(lock_path.relative_to(REPO)),
            "campaign_lock_sha256": digest(lock_bytes),
            "weights": str(weights.relative_to(REPO)),
            "weights_sha256": sha256(weights),
            "runtime": {
                "image_size": int(lock["detector"]["imgsz"]),
                "prediction_confidence_floor": 0.001,
                "operating_confidence": 0.25,
                "iou_threshold": 0.45,
                "batch_size": int(args.batch_size),
                "device": str(args.device),
            },
            "records": records_path.name,
            "records_sha256": sha256(records_path),
            "elapsed_s": elapsed,
            "implementation": str(Path(__file__).resolve().relative_to(REPO)),
            "implementation_sha256": sha256(Path(__file__).resolve()),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / ".complete").write_text(
            json.dumps({"manifest_sha256": sha256(manifest_path)}) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception:
        # Keep the incomplete directory and progress log for diagnosis; never
        # promote a partial pass to the immutable output path.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
