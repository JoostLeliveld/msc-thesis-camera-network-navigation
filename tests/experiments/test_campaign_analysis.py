"""The campaign analysis pairs runs on (task, seed) and never scores a missing cell."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def analysis():
    spec = importlib.util.spec_from_file_location("campaign_analysis", REPO / "pipeline/analyze_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(task, condition, seed, belief, outcome="goal_reached"):
    model, state = condition.rsplit("_", 1)
    value = {"task": task, "condition": condition, "model": model, "state": state, "seed": seed,
             "outcome": outcome}
    for metric in analysis().METRICS:
        value[metric] = 1.0
    value["belief_error_m"] = belief
    return value


def test_conditions_split_into_model_and_state():
    a = analysis()
    assert a.split_condition("per_camera_removal") == ("per_camera", "removal")
    assert a.split_condition("spatial_intact") == ("spatial", "intact")


def test_matched_difference_pairs_on_task_and_seed_and_drops_missing_cells():
    a = analysis()
    rows = [row("t", "spatial_intact", 1, 0.10), row("t", "spatial_removal", 1, 0.25),
            row("t", "spatial_intact", 2, 0.10), row("t", "spatial_removal", 2, 0.40),
            row("t", "spatial_intact", 3, 0.10), row("t", "spatial_removal", 3, 9.0, "infra_invalid")]
    diff = a.matched_differences(rows, "spatial_removal", "spatial_intact")["belief_error_m"]
    assert diff["n"] == 2
    assert diff["mean"] == pytest.approx(0.225)


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    a = analysis()
    first, second = a.bootstrap_ci([0.1, 0.2, 0.4, 0.3]), a.bootstrap_ci([0.1, 0.2, 0.4, 0.3])
    assert first == second
    assert first["ci95"][0] <= first["mean"] <= first["ci95"][1]
