#!/usr/bin/env python3
"""Export the frozen detector-fit/validation roles as an audited YOLO dataset."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2


ALLOWED_ROLES = {"detector_fit": "train", "detector_validation": "val"}
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def classify(
    row: dict[str, str], width: int, height: int, contract: dict
) -> tuple[str, list[str], tuple[int, int, int, int] | None, dict[str, float]]:
    pixels = int(float(row["semantic_robot_pixels"]))
    expected = tuple(finite(row, key) for key in ("expected_x0", "expected_y0", "expected_x1", "expected_y1"))
    if pixels == 0:
        if all(value is not None for value in expected):
            ex0, ey0, ex1, ey1 = expected
            outside = ex1 < 0 or ey1 < 0 or ex0 >= width or ey0 >= height
        else:
            outside = False
        reason = "negative_outside_image" if outside else "negative_fully_occluded"
        return "negative", [reason], None, {}

    coordinates = tuple(
        finite(row, key) for key in ("mask_x0", "mask_y0", "mask_x1", "mask_y1")
    )
    if any(value is None for value in coordinates) or any(value is None for value in expected):
        return "ambiguous", ["invalid_geometry"], None, {}
    x0f, y0f, x1f, y1f = coordinates
    ex0, ey0, ex1, ey1 = expected
    x0, y0, x1, y1 = map(int, coordinates)
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    expected_w = ex1 - ex0
    expected_h = ey1 - ey0
    visible_w_ratio = box_w / expected_w if expected_w > 0 else math.nan
    visible_h_ratio = box_h / expected_h if expected_h > 0 else math.nan
    bottom_v = finite(row, "mask_bottom_v")
    bottom_gap = max(0.0, ey1 - bottom_v) if bottom_v is not None else math.inf
    max_bottom_gap = max(6.0, 0.15 * expected_h) if expected_h > 0 else -math.inf
    border = int(contract["forbidden_border_contact_px"])

    reasons: list[str] = []
    if pixels < int(contract["min_semantic_area_px"]):
        reasons.append("semantic_area")
    if box_w < int(contract["min_bbox_width_px"]):
        reasons.append("bbox_width")
    if box_h < int(contract["min_bbox_height_px"]):
        reasons.append("bbox_height")
    if not math.isfinite(visible_w_ratio) or visible_w_ratio < float(contract["min_visible_to_projected_width_ratio"]):
        reasons.append("visible_width_ratio")
    if not math.isfinite(visible_h_ratio) or visible_h_ratio < float(contract["min_visible_to_projected_height_ratio"]):
        reasons.append("visible_height_ratio")
    if bottom_gap > max_bottom_gap:
        reasons.append("bottom_edge_gap")
    if x0 < border or y0 < border or x1 >= width - border or y1 >= height - border:
        reasons.append("border_contact")

    metrics = {
        "bbox_width_px": box_w,
        "bbox_height_px": box_h,
        "visible_width_ratio": visible_w_ratio,
        "visible_height_ratio": visible_h_ratio,
        "bottom_edge_gap_px": bottom_gap,
        "max_bottom_edge_gap_px": max_bottom_gap,
    }
    return ("ambiguous" if reasons else "positive"), reasons, (x0, y0, x1, y1), metrics


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_contact_sheet(entries: list[dict], capture: Path, output: Path) -> None:
    selected: dict[tuple[str, str, str], dict] = {}
    for entry in sorted(
        entries,
        key=lambda item: hashlib.sha1(
            f"{item['pose_id']}:{item['camera_id']}".encode("ascii")
        ).hexdigest(),
    ):
        key = (entry["dataset_split"], entry["camera_id"], entry["label_class"])
        selected.setdefault(key, entry)
    cards = []
    for entry in sorted(selected.values(), key=lambda item: (
        item["dataset_split"], item["label_class"], item["camera_id"]
    )):
        image = cv2.imread(str(capture / entry["source_image"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        source_height, source_width = image.shape[:2]
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        mask_coordinates = tuple(
            finite(entry, key) for key in ("mask_x0", "mask_y0", "mask_x1", "mask_y1")
        )
        if all(value is not None for value in mask_coordinates):
            x0, y0, x1, y1 = mask_coordinates
            sx, sy = 320 / source_width, 180 / source_height
            p0 = (round(x0 * sx), round(y0 * sy))
            p1 = (round(x1 * sx), round(y1 * sy))
            color = (0, 220, 0) if entry["label_class"] == "positive" else (0, 190, 255)
            cv2.rectangle(image, p0, p1, color, 2)
        title = f"{entry['dataset_split']} {entry['camera_id']} {entry['label_class']}"
        reason = entry["reasons"][:46]
        cv2.rectangle(image, (0, 0), (320, 34), (0, 0, 0), -1)
        cv2.putText(image, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, reason, (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (230, 230, 230), 1, cv2.LINE_AA)
        cards.append(image)
    columns = 5
    blank = 255 * (cards[0] * 0 + 1) if cards else None
    if not cards:
        raise RuntimeError("No audit cards were generated")
    while len(cards) % columns:
        cards.append(blank.copy())
    rows = [cv2.hconcat(cards[index:index + columns]) for index in range(0, len(cards), columns)]
    sheet = cv2.vconcat(rows)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    capture = args.capture.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset: {output}")

    manifest_path = capture / "capture_manifest.json"
    index_path = capture / "capture_index.csv"
    capture_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if capture_manifest.get("status") != "complete":
        raise RuntimeError("Master capture must be complete before label export")
    if capture_manifest.get("capture_index_sha256") != digest(index_path):
        raise RuntimeError("Master capture index hash does not match its manifest")
    if capture_manifest.get("protocol_manifest", {}).get("sha256") != digest(protocol_path):
        raise RuntimeError("Capture is not bound to the supplied frozen label protocol")

    with index_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [row for row in source_rows if row["dataset_split"] in ALLOWED_ROLES]
    if len(rows) != 4800:
        raise RuntimeError(f"Expected 4,800 detector-role views, found {len(rows)}")
    if any(row["dataset_split"] == "final_audit" for row in rows):
        raise RuntimeError("Final-audit leakage into detector export")

    for split in ALLOWED_ROLES.values():
        (output / "images" / split).mkdir(parents=True, exist_ok=False)
        (output / "labels" / split).mkdir(parents=True, exist_ok=False)
    (output / "audit").mkdir(parents=True, exist_ok=False)

    contract = protocol["detector_label_contract"]["positive_requires_all"]
    ledger: list[dict] = []
    counts: collections.Counter[tuple[str, str, str]] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        source_image = capture / row["image"]
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode {source_image}")
        height, width = image.shape[:2]
        label_class, reasons, box, metrics = classify(row, width, height, contract)
        counts[(row["dataset_split"], row["camera_id"], label_class)] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        stem = f"pose_{int(row['pose_id']):06d}_{row['camera_id']}"
        exported_image = ""
        exported_label = ""
        if label_class in {"positive", "negative"}:
            split = ALLOWED_ROLES[row["dataset_split"]]
            image_target = output / "images" / split / f"{stem}.png"
            os.link(source_image, image_target)
            label_target = output / "labels" / split / f"{stem}.txt"
            if box is None:
                label_target.write_text("", encoding="ascii")
            else:
                x0, y0, x1, y1 = box
                box_w, box_h = x1 - x0 + 1, y1 - y0 + 1
                center_x = (x0 + x1 + 1) / 2.0 / width
                center_y = (y0 + y1 + 1) / 2.0 / height
                label_target.write_text(
                    f"0 {center_x:.9f} {center_y:.9f} {box_w / width:.9f} {box_h / height:.9f}\n",
                    encoding="ascii",
                )
            exported_image = str(image_target.relative_to(output))
            exported_label = str(label_target.relative_to(output))
        ledger.append({
            "pose_id": row["pose_id"],
            "position_id": row["position_id"],
            "position_key": row["position_key"],
            "block_id": row["block_id"],
            "heading_id": row["heading_id"],
            "dataset_split": row["dataset_split"],
            "camera_id": row["camera_id"],
            "source_batch_id": row["source_batch_id"],
            "source_image": row["image"],
            "source_image_sha1": row["image_sha1"],
            "source_mask": row["robot_mask"],
            "source_mask_sha1": row["robot_mask_sha1"],
            "semantic_robot_pixels": row["semantic_robot_pixels"],
            "label_class": label_class,
            "reasons": ";".join(reasons),
            "mask_x0": row["mask_x0"], "mask_y0": row["mask_y0"],
            "mask_x1": row["mask_x1"], "mask_y1": row["mask_y1"],
            "bbox_width_px": metrics.get("bbox_width_px", ""),
            "bbox_height_px": metrics.get("bbox_height_px", ""),
            "visible_width_ratio": metrics.get("visible_width_ratio", ""),
            "visible_height_ratio": metrics.get("visible_height_ratio", ""),
            "bottom_edge_gap_px": metrics.get("bottom_edge_gap_px", ""),
            "max_bottom_edge_gap_px": metrics.get("max_bottom_edge_gap_px", ""),
            "exported_image": exported_image,
            "exported_label": exported_label,
        })

    ledger_path = output / "label_ledger.csv"
    write_csv(ledger_path, ledger, list(ledger[0]))
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n  0: warehouse_amr_blue\n",
        encoding="utf-8",
    )
    sheet_path = output / "audit" / "label_contact_sheet.jpg"
    make_contact_sheet(ledger, capture, sheet_path)

    summary = {
        "schema": "thesis_detector_dataset.v1",
        "status": "exported_pending_visual_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_id": protocol["identity"]["pipeline_id"],
        "allowed_roles": sorted(ALLOWED_ROLES),
        "final_audit_accessed": False,
        "capture": str(capture),
        "counts": [
            {"role": role, "camera": camera, "class": kind, "views": count}
            for (role, camera, kind), count in sorted(counts.items())
        ],
        "reason_counts": dict(sorted(reason_counts.items())),
        "totals": dict(collections.Counter(item["label_class"] for item in ledger)),
        "identities": {
            "capture_manifest_sha256": digest(manifest_path),
            "capture_index_sha256": digest(index_path),
            "label_protocol_sha256": digest(protocol_path),
            "export_script_sha256": digest(Path(__file__).resolve()),
            "label_ledger_sha256": digest(ledger_path),
            "dataset_yaml_sha256": digest(yaml_path),
            "contact_sheet_sha256": digest(sheet_path),
        },
    }
    summary_path = output / "dataset_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
