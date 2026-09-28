#!/usr/bin/env python3
"""Apply the frozen cross-resolution selection rule to Stage-05 reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-bin-support", type=int, default=20)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol_hash = sha256(protocol_path)
    expected = {int(item["imgsz"]) for item in json.loads(protocol_path.read_text())["resolution_trials"]}
    candidates = []
    for path_arg in args.reports:
        path = path_arg.resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["protocol_sha256"] != protocol_hash:
            raise RuntimeError(f"Protocol hash mismatch in {path}")
        if report["population"] != "detector_validation only; 100% real; ambiguous excluded":
            raise RuntimeError(f"Unauthorized evaluation population in {path}")
        point = report["primary_fixed_operating_point"]
        bins = point["recall_by_visible_width_px"]
        sufficiently_populated = {
            key: value for key, value in bins.items()
            if int(value["positives"]) >= args.minimum_bin_support
        }
        if not sufficiently_populated:
            raise RuntimeError(f"No sufficiently populated width bins in {path}")
        candidates.append({
            "imgsz": int(report["imgsz"]), "report": str(path), "report_sha256": sha256(path),
            "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"],
            "minimum_width_bin_recall": min(x["recall"] for x in sufficiently_populated.values()),
            "false_detections_per_negative_image": point["false_detections_per_negative_image"],
            "precision_at_0.25": point["precision"], "recall_at_0.25": point["recall"],
            "ap50": report["ap50"], "ap50_95": report["ap50_95"],
            "width_bins": sufficiently_populated,
        })
    if {x["imgsz"] for x in candidates} != expected:
        raise RuntimeError(f"Expected resolutions {sorted(expected)}, got {sorted(x['imgsz'] for x in candidates)}")

    best_minimum = max(x["minimum_width_bin_recall"] for x in candidates)
    within_two_points = [x for x in candidates if x["minimum_width_bin_recall"] >= best_minimum - 0.02]
    winner = min(
        within_two_points,
        key=lambda x: (x["false_detections_per_negative_image"], -x["minimum_width_bin_recall"], x["imgsz"]),
    )
    decision = {
        "schema": "thesis_detector_resolution_selection.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "selected_pending_stage05_lock",
        "protocol": str(protocol_path), "protocol_sha256": protocol_hash,
        "minimum_bin_support": args.minimum_bin_support,
        "rule": "Maximize minimum sufficiently populated width-bin recall at confidence 0.25; among candidates within 0.02, minimize false detections per negative; then prefer greater minimum recall and lower resolution.",
        "candidates": sorted(candidates, key=lambda x: x["imgsz"]),
        "selected": winner,
        "commissioning_fit_accessed": False,
        "final_audit_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
