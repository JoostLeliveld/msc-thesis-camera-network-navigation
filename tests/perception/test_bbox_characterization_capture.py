from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pipeline" / "capture" / "capture_positions.py"
SPEC = importlib.util.spec_from_file_location("capture_bbox_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def pair(stamp_ns: int):
    return MODULE.Pair(np.zeros((1, 1, 3), dtype=np.uint8), None, stamp_ns, None, np.nan)


def test_closest_timestamp_batch_does_not_merge_adjacent_rounds():
    candidates = {
        "A": [pair(1_000_000_000), pair(1_200_000_000)],
        "B": [pair(1_000_000_000), pair(1_200_000_000)],
        # C's latest callback is already from the following round. Independent latest-frame
        # selection would produce a 0.2 s span; joint selection must recover the 1.2 s round.
        "C": [pair(1_200_000_000), pair(1_400_000_000)],
    }
    selected = MODULE._closest_timestamp_batch(candidates, max_span_s=0.05)
    assert selected is not None
    assert {item.image_stamp_ns for item in selected.values()} == {1_200_000_000}


def test_closest_timestamp_batch_rejects_when_no_coherent_round_exists():
    candidates = {
        "A": [pair(1_000_000_000)],
        "B": [pair(1_200_000_000)],
    }
    assert MODULE._closest_timestamp_batch(candidates, max_span_s=0.05) is None
