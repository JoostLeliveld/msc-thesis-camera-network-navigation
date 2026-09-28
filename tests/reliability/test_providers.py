from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    ContractValidationError,
    FixedCameraReliabilityProvider,
    GridMapReliabilityProvider,
    MultiCameraReliabilityProvider,
    quality_from_gp_artifact,
)


def _write_npz(path: Path) -> None:
    np = pytest.importorskip("numpy")
    np.savez(
        path,
        xs=np.asarray([0.0, 1.0]),
        ys=np.asarray([0.0, 1.0]),
        P_conservative_plan_map=np.asarray([[0.2, 0.4], [0.6, 0.8]]),
        F_std_map=np.asarray([[0.0, 0.1], [0.2, 0.3]]),
    )


def test_fixed_provider_returns_camera_quality() -> None:
    provider = FixedCameraReliabilityProvider(camera_id="camera_A", p_available=0.7)
    quality = provider.query("camera_A", (1.0, 2.0), timestamp_s=3.0)

    assert quality.camera_id == "camera_A"
    assert quality.p_available == 0.7
    assert quality.source_model == "fixed_camera_reliability"
    with pytest.raises(ContractValidationError, match="camera_B"):
        provider.query("camera_B", (1.0, 2.0))


def test_grid_provider_bilinear_interpolates_gp_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "visibility_gp.npz"
    _write_npz(artifact)

    provider = GridMapReliabilityProvider.from_npz(artifact, camera_id="camera_A")
    quality = provider.query("camera_A", (0.5, 0.5), timestamp_s=1.0)

    assert quality.p_available == pytest.approx(0.5)
    assert quality.epistemic_score == pytest.approx(0.15)
    assert quality.conditional_cov_uv[0][0] > 0.0
    assert quality.conditional_cov_uv[0][1] == 0.0
    assert provider.contains_xy((0.5, 0.5))
    assert provider.artifact_sha256.startswith("sha256:")


def test_grid_provider_support_policies(tmp_path: Path) -> None:
    artifact = tmp_path / "visibility_gp.npz"
    _write_npz(artifact)

    conservative = GridMapReliabilityProvider.from_npz(
        artifact,
        camera_id="camera_A",
        min_probability=0.05,
        out_of_bounds_policy="min",
    )
    assert conservative.query("camera_A", (5.0, 5.0)).p_available == pytest.approx(0.05)
    assert conservative.query("camera_A", (5.0, 5.0)).epistemic_score == pytest.approx(1.0)

    clamped = GridMapReliabilityProvider.from_npz(
        artifact,
        camera_id="camera_A",
        out_of_bounds_policy="clamp",
    )
    assert clamped.query("camera_A", (5.0, 5.0)).p_available == pytest.approx(0.8)

    strict = GridMapReliabilityProvider.from_npz(
        artifact,
        camera_id="camera_A",
        out_of_bounds_policy="raise",
    )
    with pytest.raises(ContractValidationError, match="outside artifact support"):
        strict.query("camera_A", (5.0, 5.0))


def test_multicamera_provider_dispatches_by_camera(tmp_path: Path) -> None:
    artifact = tmp_path / "visibility_gp.npz"
    _write_npz(artifact)
    providers = MultiCameraReliabilityProvider(
        {
            "camera_A": FixedCameraReliabilityProvider(camera_id="camera_A", p_available=0.3),
            "camera_B": GridMapReliabilityProvider.from_npz(artifact, camera_id="camera_B"),
        }
    )

    qualities = providers.query_all((0.5, 0.5), timestamp_s=1.0)

    assert qualities["camera_A"].p_available == pytest.approx(0.3)
    assert qualities["camera_B"].p_available == pytest.approx(0.5)
    with pytest.raises(ContractValidationError, match="Unknown camera_id"):
        providers.query("camera_C", (0.0, 0.0))


def test_quality_from_gp_artifact_convenience(tmp_path: Path) -> None:
    artifact = tmp_path / "visibility_gp.npz"
    _write_npz(artifact)

    quality = quality_from_gp_artifact(
        artifact,
        camera_id="camera_A",
        belief_xy=(0.0, 1.0),
    )

    assert quality.camera_id == "camera_A"
    assert quality.p_available == pytest.approx(0.6)


def test_current_gp_artifact_loads_if_present() -> None:
    artifact = ROOT / "paper_artifacts" / "gp" / "warehouse_visibility_gp_v1" / "yolo_score_raw_gp.npz"
    if not artifact.is_file():
        pytest.skip("current GP artifact is not present in this checkout")

    provider = GridMapReliabilityProvider.from_npz(artifact, camera_id="camera_A", out_of_bounds_policy="clamp")
    x_mid = 0.5 * (provider.x_min + provider.x_max)
    y_mid = 0.5 * (provider.y_min + provider.y_max)
    quality = provider.query("camera_A", (x_mid, y_mid), timestamp_s=0.0)

    assert 0.0 < quality.p_available < 1.0
    assert quality.conditional_cov_uv[0][0] > 0.0
