from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

import reliability  # noqa: E402


PAPER_EXTENSION_PUBLIC_API = {
    "CalibrationPoint",
    "ToroCovarianceModel",
    "bin_observations",
    "constant_velocity_predict",
    "CalibrationError",
    "IsotonicCalibrator",
    "LogisticCalibrator",
    "MultivariateLogisticCalibrator",
    "reliability_curve",
    "TrustFeatures",
    "TrustStacker",
    "split_groups",
    "CovarianceEstimate",
    "default_shrinkage_lambda",
    "estimate_conditional_covariance",
    "matrix_nll",
    "chi2_coverage",
    "sharpness",
    "InnovationHealthConfig",
    "InnovationHealthMonitor",
    "CalibrationHealthState",
    "HealthDebouncerConfig",
    "HealthDebouncer",
    "isolate_suspect_camera",
    "FuseOrSelectDecision",
    "joseph_update_2d",
    "robust_reweight_covariance",
    "expected_information_gain",
    "select_information_best",
    "fuse_or_select",
    "plan_reliability",
    "plan_covariance",
    "expected_information_update",
    "sequential_expected_update",
    "batch_plan_query",
}


def test_paper_extension_symbols_are_available_from_package_root() -> None:
    missing_attributes = sorted(
        name for name in PAPER_EXTENSION_PUBLIC_API if not hasattr(reliability, name)
    )
    missing_exports = sorted(
        name for name in PAPER_EXTENSION_PUBLIC_API if name not in reliability.__all__
    )

    assert missing_attributes == []
    assert missing_exports == []


def test_package_export_list_has_no_duplicates() -> None:
    assert len(reliability.__all__) == len(set(reliability.__all__))
