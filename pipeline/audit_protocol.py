#!/usr/bin/env python3
"""Freeze the final-audit protocol after the refit: every locked input by path and hash.

The audit population is the final_audit role of the dataset lock (the v5 audit set). The
D_dev evaluation manifest is locked too, because it carries the R_proj sigma_px fitted on
D_R. Writing this file is the last step before final_audit is opened; it refuses to run
while any locked input is missing, and once the audit has been opened.

    python3 pipeline/audit_protocol.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V8 = "logs/thesis/fits"
LOCK = "pipeline/dataset_lock.json"


def sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def main() -> int:
    protocol = {
        "schema": "thesis_reference_final_audit_protocol.v1",
        "authorized_role": "final_audit",
        "method_authority": "docs/METHOD.md",
        "prohibitions": [
            "No fitting, calibration, threshold selection or model selection uses final_audit.",
            "Every reported aggregate uses complete physical position as its independent unit.",
            "The audit is run once against the frozen detector, gate, correction, covariance and planning artifacts.",
        ],
    }
    lock = json.loads((REPO / LOCK).read_text())
    expected = lock["opportunity_accounting"]["expected_final_audit_opportunities"]
    protocol["population"] = {
        "positions": lock["partition"]["roles"]["final_audit"],
        "opportunities": expected,
        "merge_loader": "pipeline/dataset.py",
    }
    inputs = {
        "campaign_lock": LOCK,
        "gate_config": "config/sensor_gate.yaml",
        "label_protocol": "pipeline/detector/label_protocol.json",
        "detector": lock["detector"]["checkpoint"],
        "runtime_R0": f"{V8}/runtime_r012/R0_global_full.json",
        "runtime_R1": f"{V8}/runtime_r012/R1_per_camera_full.json",
        "runtime_R2": f"{V8}/runtime_r012/R2_spatial_residual.json",
        "planning_M0": f"{V8}/planning_precision/m0_planning_precision.npz",
        "planning_M1": f"{V8}/planning_precision/m1_planning_precision.npz",
        "planning_M2": f"{V8}/planning_precision/m2_planning_precision.npz",
        "ddev_evaluation": f"{V8}/ddev_evaluation/manifest.json",
        "implementation": "pipeline/final_audit.py",
    }
    protocol["locked_inputs"] = {key: {"path": rel, "sha256": sha(rel)}
                                 for key, rel in inputs.items()}
    protocol["amendment"] = "docs/METHOD.md, Amendment 2026-09-23/24"
    protocol["status"] = "frozen_before_final_audit_access"
    out = REPO / "logs/thesis/final_audit_protocol.json"
    audit = REPO / "logs/thesis/final_audit"
    # The protocol may be rewritten until the audit is opened, never after.
    if audit.exists() or audit.with_name(audit.name + ".incomplete").exists():
        raise FileExistsError(f"final audit already opened: {audit}")
    out.write_text(json.dumps(protocol, indent=2) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
