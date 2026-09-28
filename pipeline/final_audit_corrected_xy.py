#!/usr/bin/env python3
"""Rebuild the corrected positions of the admitted final-audit observations.

final_audit.py stores only the corrected error norm. This re-applies the locked runtime
correction to the stored detector box, confidence and raw projection (no detector run, no
fitting) and refuses to write unless every rebuilt error equals the stored one.

    python3 pipeline/final_audit_corrected_xy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src/reliability"), str(REPO / "src/unav_common")]

from pipeline.dataset import image_path, load_rows  # noqa: E402
from reliability.commissioned_visibility import CommissionedVisibilitySensorModel  # noqa: E402
from unav_common.visibility_patch import visibility_grid_from_bgr_frame  # noqa: E402

AUDIT = REPO / "logs/thesis/final_audit"
PROTOCOL = REPO / "logs/thesis/final_audit_protocol.json"
OUTPUT = REPO / "logs/thesis/final_audit_corrected_xy.npz"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    # final_audit.py takes the corrected estimate from the first model; the correction is shared.
    model = CommissionedVisibilitySensorModel(REPO / protocol["locked_inputs"]["runtime_R0"]["path"])
    capture = {(r["position_key"], int(r["yaw_idx"]), r["camera_id"]): r
               for r in load_rows() if r["stratum"] == "final_audit"}
    rows = [json.loads(line) for line in (AUDIT / "evaluated.jsonl").open()]
    rows = [r for r in rows if r["outcome"] == "admitted"]
    keys, cams, truth, raw, cor = [], [], [], [], []
    for r in rows:
        row = capture[(r["position_key"], r["yaw_idx"], r["camera_id"])]
        if row["image_sha1"] != r["image_sha1"]:
            raise RuntimeError(f"image drift at {r['position_key']}")
        image = cv2.imread(str(image_path(row)), cv2.IMREAD_COLOR)
        grid = visibility_grid_from_bgr_frame(image, r["bbox_xyxy"])
        estimate, _ = model.correct_and_covariance(
            r["camera_id"], np.asarray(r["raw_xy_m"]), r["bbox_xyxy"], float(r["confidence"]),
            grid, float(row["robot_yaw"]))
        estimate = np.asarray(estimate, dtype=float)
        error = float(np.linalg.norm(estimate - np.asarray(r["truth_xy_m"])))
        if abs(error - r["corrected_error_m"]) > 1e-9:
            raise RuntimeError(f"{r['position_key']} {r['camera_id']}: rebuilt {error} != audit {r['corrected_error_m']}")
        keys.append(r["position_key"]); cams.append(r["camera_id"])
        truth.append(r["truth_xy_m"]); raw.append(r["raw_xy_m"]); cor.append(estimate)
    np.savez(OUTPUT, position_key=np.asarray(keys), camera=np.asarray(cams),
             truth_xy_m=np.asarray(truth), raw_xy_m=np.asarray(raw), corrected_xy_m=np.asarray(cor),
             camera_xy_m=np.asarray([model.camera_xy[c] for c in cams]))
    print(OUTPUT, len(keys), "observations", len(set(keys)), "positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
