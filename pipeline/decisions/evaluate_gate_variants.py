#!/usr/bin/env python3
"""Choose the sensor-gate edge rule on D_dev, by a rule fixed before any result (2026-09-23).

Variants (gate_variants/gate_G*.yaml), each refitted from the same detector inference:
  G0  full_bbox,      5 px   (current)
  G1  selected_pixel, 5 px   (only the projected bottom-centre must be inside the image)
  G2  selected_pixel, 1 px
  G3  no edge or size check (confidence and ground projection only)

Selection: the most permissive variant such that, on D_dev,
  (1) corrected RMSE on the observations G0 admits rises by at most 0.1 cm, and
  (2) the observations it admits beyond G0 have a corrected 95th-percentile error no
      worse than G0's own.
If no candidate passes, G0 stays.

    python3 pipeline/decisions/evaluate_gate_variants.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V = REPO / "logs/thesis/evidence/gate_variants"
ORDER = ("G0", "G1", "G2", "G3")
RULE = ("most permissive variant with (1) corrected D_dev RMSE on G0-admitted observations "
        "within +0.1 cm of G0 and (2) extra observations' corrected p95 no worse than G0's")


def dev_residuals(name: str) -> dict:
    root = V / name
    manifest = json.loads((root / "correction/manifest.json").read_text())
    z = np.load(root / "correction" / manifest["artifacts"]["predictions"]["path"])
    out = {}
    for i in np.flatnonzero(z["role"] == "D_dev"):
        key = (int(z["plan_pose_index"][i]), str(z["camera"][i]))
        out[key] = float(np.linalg.norm(z["residual_world_m"][i]))
    return out


def coverage(name: str) -> dict:
    admitted = collections.defaultdict(set)
    poses = set()
    with (V / name / "gate_dataset/opportunities.jsonl").open() as handle:
        for line in handle:
            o = json.loads(line)
            pose = (o["position_key"], o["yaw_idx"])
            poses.add(pose)
            if o["opportunity_outcome"] == "admitted":
                admitted[pose].add(o["camera_id"])
    per_camera = collections.Counter(c for cams in admitted.values() for c in cams)
    return {"admitted_by_camera": dict(sorted(per_camera.items())),
            "poses_with_2plus_cameras": sum(len(v) >= 2 for v in admitted.values()) / len(poses)}


def rmse(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def main() -> int:
    res = {name: dev_residuals(name) for name in ORDER}
    base_keys = set(res["G0"])
    base_p95 = float(np.quantile(list(res["G0"].values()), 0.95))
    report = {"rule": RULE, "variants": {}}
    chosen = "G0"
    for name in ORDER:
        r = res[name]
        common = base_keys & set(r)
        extra = set(r) - base_keys
        row = {
            "D_dev_admitted": len(r),
            "rmse_on_G0_admitted_cm": 100 * rmse([r[k] for k in common]),
            "G0_rmse_on_same_cm": 100 * rmse([res["G0"][k] for k in common]),
            "extra_admitted": len(extra),
            "extra_p95_cm": 100 * float(np.quantile([r[k] for k in extra], 0.95)) if extra else None,
            "G0_p95_cm": 100 * base_p95,
            **coverage(name),
        }
        row["passes"] = bool(name == "G0" or (
            row["rmse_on_G0_admitted_cm"] - row["G0_rmse_on_same_cm"] <= 0.1
            and (row["extra_p95_cm"] is None or row["extra_p95_cm"] <= row["G0_p95_cm"])))
        report["variants"][name] = row
        if row["passes"]:
            chosen = name
    report["chosen"] = chosen
    (V / "gate_variant_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, row in report["variants"].items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
