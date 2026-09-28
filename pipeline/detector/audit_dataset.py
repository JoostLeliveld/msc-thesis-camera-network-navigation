#!/usr/bin/env python3
"""Audit the Stage-04 YOLO export without reading commissioning/final-audit views."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROLES = {"detector_fit": "train", "detector_validation": "val"}
CLASSES = {"positive", "negative", "ambiguous"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def reason_sheet(rows: list[dict[str, str]], capture: Path, output: Path) -> int:
    candidates: dict[tuple[str, str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        if row["label_class"] != "ambiguous":
            continue
        for reason in row["reasons"].split(";"):
            if reason:
                candidates[(row["dataset_split"], row["camera_id"], reason)].append(row)
    selected = []
    for key, options in sorted(candidates.items()):
        options.sort(key=lambda row: hashlib.sha1(
            f"{row['pose_id']}:{row['camera_id']}:{key[2]}".encode("ascii")
        ).hexdigest())
        selected.append((key, options[0]))

    cards = []
    for (role, camera, reason), row in selected:
        source = cv2.imread(str(capture / row["source_image"]), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"Cannot decode {row['source_image']}")
        height, width = source.shape[:2]
        card = cv2.resize(source, (320, 180), interpolation=cv2.INTER_AREA)
        coordinates = [number(row[key]) for key in ("mask_x0", "mask_y0", "mask_x1", "mask_y1")]
        if all(value is not None for value in coordinates):
            x0, y0, x1, y1 = coordinates
            cv2.rectangle(
                card,
                (round(x0 * 320 / width), round(y0 * 180 / height)),
                (round(x1 * 320 / width), round(y1 * 180 / height)),
                (0, 190, 255),
                2,
            )
        cv2.rectangle(card, (0, 0), (320, 35), (0, 0, 0), -1)
        cv2.putText(card, f"{role} {camera}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.37, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(card, reason, (4, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 210, 255), 1, cv2.LINE_AA)
        cards.append(card)
    if not cards:
        raise RuntimeError("No ambiguous-reason audit samples found")
    blank = np.full_like(cards[0], 255)
    while len(cards) % 5:
        cards.append(blank.copy())
    sheet = cv2.vconcat([
        cv2.hconcat(cards[index:index + 5]) for index in range(0, len(cards), 5)
    ])
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write {output}")
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    capture = args.capture.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    ledger_path = dataset / "label_ledger.csv"
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    checks: dict[str, bool] = {}
    checks["exact_ledger_rows"] = len(rows) == 4800
    checks["roles_limited_to_detector"] = set(row["dataset_split"] for row in rows) == set(ROLES)
    checks["class_names_valid"] = set(row["label_class"] for row in rows) <= CLASSES
    checks["final_audit_not_accessed"] = manifest.get("final_audit_accessed") is False
    checks["ledger_identity"] = manifest["identities"]["label_ledger_sha256"] == sha256(ledger_path)
    checks["protocol_identity"] = manifest["identities"]["label_protocol_sha256"] == sha256(protocol_path)
    checks["capture_identity"] = (
        manifest["identities"]["capture_manifest_sha256"] == sha256(capture / "capture_manifest.json")
        and manifest["identities"]["capture_index_sha256"] == sha256(capture / "capture_index.csv")
    )

    memberships: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        memberships[row["position_id"]].add(row["dataset_split"])
    checks["position_roles_disjoint"] = all(len(roles) == 1 for roles in memberships.values())
    checks["exact_position_counts"] = (
        len({row["position_id"] for row in rows if row["dataset_split"] == "detector_fit"}) == 80
        and len({row["position_id"] for row in rows if row["dataset_split"] == "detector_validation"}) == 40
    )

    failures = []
    for row in rows:
        exported = row["label_class"] in {"positive", "negative"}
        if exported != bool(row["exported_image"] and row["exported_label"]):
            failures.append(f"export-accounting:{row['pose_id']}:{row['camera_id']}")
            continue
        if not exported:
            continue
        image_path = dataset / row["exported_image"]
        label_path = dataset / row["exported_label"]
        if not image_path.is_file() or not label_path.is_file():
            failures.append(f"missing:{row['pose_id']}:{row['camera_id']}")
            continue
        lines = [line for line in label_path.read_text(encoding="ascii").splitlines() if line]
        if row["label_class"] == "negative":
            if lines:
                failures.append(f"negative-has-label:{row['pose_id']}:{row['camera_id']}")
        else:
            if len(lines) != 1:
                failures.append(f"positive-label-count:{row['pose_id']}:{row['camera_id']}")
                continue
            fields = lines[0].split()
            if len(fields) != 5 or fields[0] != "0":
                failures.append(f"positive-label-schema:{row['pose_id']}:{row['camera_id']}")
                continue
            values = [float(value) for value in fields[1:]]
            if not all(0 < value <= 1 for value in values):
                failures.append(f"positive-label-bounds:{row['pose_id']}:{row['camera_id']}")
    checks["exported_files_and_labels_valid"] = not failures

    totals = collections.Counter(row["label_class"] for row in rows)
    checks["both_splits_have_all_classes"] = all(
        all(any(row["dataset_split"] == role and row["label_class"] == kind for row in rows)
            for kind in CLASSES)
        for role in ROLES
    )
    checks["training_has_substantial_positive_and_negative_support"] = (
        sum(row["dataset_split"] == "detector_fit" and row["label_class"] == "positive" for row in rows) >= 1000
        and sum(row["dataset_split"] == "detector_fit" and row["label_class"] == "negative" for row in rows) >= 1000
    )
    checks["validation_has_substantial_positive_and_negative_support"] = (
        sum(row["dataset_split"] == "detector_validation" and row["label_class"] == "positive" for row in rows) >= 500
        and sum(row["dataset_split"] == "detector_validation" and row["label_class"] == "negative" for row in rows) >= 500
    )

    reason_sheet_path = dataset / "audit" / "ambiguous_reason_contact_sheet.jpg"
    reason_examples = reason_sheet(rows, capture, reason_sheet_path)
    report = {
        "schema": "thesis_detector_dataset_audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(checks.values()),
        "totals": dict(totals),
        "eligible_training_images": totals["positive"] + totals["negative"],
        "ambiguous_reason_examples": reason_examples,
        "label_failures": failures[:100],
        "identities": {
            "dataset_manifest_sha256": sha256(manifest_path),
            "label_ledger_sha256": sha256(ledger_path),
            "ambiguous_reason_contact_sheet_sha256": sha256(reason_sheet_path),
            "audit_script_sha256": sha256(Path(__file__).resolve()),
        },
        "sealed_data_statement": (
            "Only detector_fit and detector_validation ledger rows and source images were read; "
            "commissioning_fit and final_audit were not accessed."
        ),
    }
    output = dataset / "dataset_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
