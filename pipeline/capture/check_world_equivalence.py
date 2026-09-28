#!/usr/bin/env python3
"""Check that today's world renders the same images as the world v8 was captured in.

The contact sensors were removed from the world file after v8; nothing that renders
changed but one pixel. This re-captures 12 v8 top-up poses (3 positions x 4 headings; the
third position is one of the 12 v8 positions inside an object, which the capture's pose
check skips) with the current world and compares every camera image with its v8 image.
Measured 2026-09-24: every image is identical except one static background pixel of
camera B, (x=170, y=136), in every camera B image whatever the robot pose. The check
passes only if that is the whole difference; the dataset audit then accepts both world
hashes.

    python3 pipeline/capture/check_world_equivalence.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
V8 = REPO / "logs/thesis/captures/v8/topup"
NEW = REPO / "logs/thesis/captures/v9/world_equivalence/capture"
OUT = REPO / "logs/thesis/captures/v9/world_equivalence/report.json"
KNOWN_PIXEL = [170, 136]


def index(directory):
    with (directory / "capture_index.csv").open(newline="") as handle:
        return {(int(r["pose_id"]), r["camera_id"]): r for r in csv.DictReader(handle)
                if r["capture_status"] == "ok"}


def main() -> int:
    old, new = index(V8), index(NEW)
    rows, worst = [], 0
    for key, r in sorted(new.items()):
        a = cv2.imread(str(V8 / old[key]["image"]), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(NEW / r["image"]), cv2.IMREAD_UNCHANGED)
        diff = int(np.abs(a.astype(int) - b.astype(int)).max())
        worst = max(worst, diff)
        ys, xs = np.nonzero((a != b).any(axis=-1))
        rows.append({"pose_id": key[0], "camera": key[1], "identical_sha1": old[key]["image_sha1"] == r["image_sha1"],
                     "max_abs_pixel_difference": diff, "differing_pixels": [[int(x), int(y)] for x, y in zip(xs, ys)]})
    m_old = json.loads((V8 / "capture_manifest.json").read_text())
    m_new = json.loads((NEW / "capture_manifest.json").read_text())
    report = {"compared_images": len(rows), "identical_images": sum(r["identical_sha1"] for r in rows),
              "max_abs_pixel_difference": worst,
              "declared_exception": {"camera": "camera_B", "pixel_xy": KNOWN_PIXEL},
              "passed": len(rows) >= 40 and all(
                  not r["differing_pixels"] or (r["camera"] == "camera_B" and r["differing_pixels"] == [KNOWN_PIXEL])
                  for r in rows),
              "v8_world_sha256": m_old.get("world_sha256"), "current_world_sha256": m_new.get("world_sha256"),
              "images": rows}
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
