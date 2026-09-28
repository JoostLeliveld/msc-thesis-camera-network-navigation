"""The learned box correction as a runtime observation model.

The point of these tests is that the deployed model and the offline replay are the same
function. A drift between them would make the closed-loop arm incomparable to the offline
column it is meant to be judged against, and nothing in a drive log would reveal it.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src/reliability'))
sys.path.insert(0, str(REPO / 'experiments/camera_observation_characterization'))

ARTIFACT = (REPO / 'logs/perception_models/box_feature_bias_correction_20260831'
            / 'models.joblib')
CAPTURE = (REPO / 'logs/perception_datasets/warehouse_v2_bbox_characterization_20260831'
           / 'bias_update_interpretations.csv')

pytestmark = pytest.mark.skipif(
    not ARTIFACT.is_file(), reason='packaged box-correction artifact not present')


def _model():
    from reliability.learned_box_correction import LearnedBoxCorrection
    return LearnedBoxCorrection(ARTIFACT)


def test_artifact_declares_the_expected_feature_contract():
    """Loading must fail loudly if the artifact's feature order ever changes."""
    model = _model()
    assert model.camera_ids == ['camera_A', 'camera_B', 'camera_C', 'camera_D', 'camera_E']
    assert 'along/across' in model.target


def test_rejects_a_foreign_schema(tmp_path):
    import joblib

    from reliability.learned_box_correction import LearnedBoxCorrection

    payload = joblib.load(ARTIFACT)
    payload['schema'] = 'something_else.v9'
    path = tmp_path / 'foreign.joblib'
    joblib.dump(payload, path)
    with pytest.raises(ValueError, match='schema'):
        LearnedBoxCorrection(path)


def test_rejects_a_permuted_feature_order(tmp_path):
    """A silently reordered feature vector is the failure this guard exists for."""
    import joblib

    from reliability.learned_box_correction import LearnedBoxCorrection

    payload = joblib.load(ARTIFACT)
    names = list(payload['feature_names'])
    names[0], names[1] = names[1], names[0]
    payload['feature_names'] = names
    path = tmp_path / 'permuted.joblib'
    joblib.dump(payload, path)
    with pytest.raises(ValueError, match='feature order'):
        LearnedBoxCorrection(path)


def test_missing_box_is_refused_not_passed_through():
    """No box means no correction, and the caller must be told rather than guessing."""
    model = _model()
    assert model.correct('camera_B', (0.0, -5.0), None, 0.9) is None
    assert model.correct('camera_B', (0.0, -5.0), (1.0, 2.0), 0.9) is None


def test_unknown_camera_is_refused():
    model = _model()
    assert model.correct('camera_Z', (0.0, -5.0), (600.0, 400.0, 660.0, 440.0), 0.9) is None


@pytest.mark.skipif(not CAPTURE.is_file(), reason='characterization capture not present')
def test_runtime_matches_the_offline_replay_exactly():
    """The deployed correction must be the same function as the offline one.

    Both paths are run on real capture rows and required to agree to floating-point
    noise. If they diverge, the closed-loop arm is measuring a different model than the
    offline column it is compared against.
    """
    import joblib

    from fit_bias_updates import apply_correction, features as offline_features

    payload = joblib.load(ARTIFACT)
    geometry = payload['camera_geometry']
    network = payload['neural_model']
    cameras = payload['camera_ids']
    model = _model()

    with CAPTURE.open(encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row['raw_valid'] == '1'][:300]
    assert rows, 'no usable rows in the capture'

    worst = 0.0
    compared = 0
    for row in rows:
        geom = geometry[row['camera_id']]
        one_hot = [1.0 if row['camera_id'] == name else 0.0 for name in cameras]
        offline = network.predict([list(offline_features(row, geom)) + one_hot])[0]
        expected = apply_correction(row, geom, offline)

        got = model.correct(
            row['camera_id'],
            (float(row['raw_x']), float(row['raw_y'])),
            (float(row['x0']), float(row['y0']), float(row['x1']), float(row['y1'])),
            float(row['confidence']))
        assert got is not None
        worst = max(worst, math.hypot(got[0] - expected[0], got[1] - expected[1]))
        compared += 1

    assert compared >= 100
    assert worst < 1e-9, f'runtime and offline disagree by {worst:.3e} m'


def test_correction_moves_the_reading_a_plausible_distance():
    """The raw box reads about 30 cm short, so a sane correction is tens of centimetres."""
    model = _model()
    raw = (0.0, -5.0)
    got = model.correct('camera_B', raw, (600.0, 400.0, 660.0, 440.0), 0.93)
    assert got is not None
    moved = math.hypot(got[0] - raw[0], got[1] - raw[1])
    assert 0.01 < moved < 1.0, f'correction of {moved:.3f} m is not physically plausible'
