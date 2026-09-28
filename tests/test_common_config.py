"""Shared configuration contracts must reject silent model changes."""

import pytest
import numpy as np

from unav_common.config import parse_bev_affine_calibration, parse_bool


@pytest.mark.parametrize("value", [False, 0, 0.0, np.bool_(False), np.int64(0), "false", "OFF", "n"])
def test_parse_bool_preserves_explicit_false(value):
    assert parse_bool(value) is False


@pytest.mark.parametrize("value", [True, 1, 1.0, np.bool_(True), np.int64(1), "true", "ON", "y"])
def test_parse_bool_preserves_explicit_true(value):
    assert parse_bool(value) is True


@pytest.mark.parametrize("value", ["fasle", "", 2, -1, None, [], {}])
def test_parse_bool_rejects_ambiguous_truthiness(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        parse_bool(value)


def test_empty_bev_affine_explicitly_selects_constant_offset_model():
    assert parse_bev_affine_calibration("") is None


def test_bev_affine_accepts_exact_finite_six_tuple():
    assert parse_bev_affine_calibration("1, 0, 0; 0, 1, .127") == (
        1.0, 0.0, 0.0, 0.0, 1.0, 0.127,
    )


@pytest.mark.parametrize(
    "value",
    [0, "1,2,3", "1,2,3,4,5,", "1,2,x,4,5,6", "1,2,3,4,5,nan"],
)
def test_malformed_bev_affine_cannot_fall_back_to_another_model(value):
    with pytest.raises(ValueError, match="bev_affine_calibration"):
        parse_bev_affine_calibration(value)
