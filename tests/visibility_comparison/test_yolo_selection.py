from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/perception',):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from perception.core.yolo_selection import collect_candidate_detections, select_best_detection


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._values, dtype=float)


class _Boxes:
    def __init__(self, cls: list[int], conf: list[float], xyxy: list[list[float]]):
        self.cls = _Tensor(cls)
        self.conf = _Tensor(conf)
        self.xyxy = _Tensor(xyxy)

    def __len__(self):
        return int(self.conf.numpy().shape[0])


class _Result:
    def __init__(self):
        self.boxes = _Boxes(
            cls=[0],
            conf=[0.18],
            xyxy=[[10.0, 20.0, 30.0, 60.0]],
        )
        self.masks = None


class _Masks:
    def __init__(self, polygons):
        self.xy = polygons


class _MaskedResult:
    def __init__(self):
        self.boxes = _Boxes(
            cls=[0],
            conf=[0.70],
            xyxy=[[10.0, 20.0, 30.0, 60.0]],
        )
        self.masks = _Masks([
            np.array(
                [
                    [12.0, 22.0],
                    [28.0, 22.0],
                    [28.0, 58.0],
                    [18.0, 60.0],
                    [12.0, 58.0],
                ],
                dtype=float,
            )
        ])


def test_select_best_detection_preserves_subthreshold_raw_score() -> None:
    selected = select_best_detection(
        _Result(),
        confidence_threshold=0.25,
        target_ids={0},
        use_masks=True,
        mask_min_area=0.0,
        mask_bottom_band_px=3,
    )

    assert selected['detected'] is False
    assert selected['detected_after_threshold'] is False
    assert selected['raw_best_score'] == 0.18
    assert selected['selected_score'] == 0.18
    assert selected['best_class_id'] == 0
    assert selected['selected_pixel_source'] == 'bbox_bottom'
    assert selected['selected_u'] == 20.0
    assert selected['selected_v'] == 60.0


def test_select_best_detection_keeps_bbox_bottom_as_state_point_when_mask_exists() -> None:
    selected = select_best_detection(
        _MaskedResult(),
        confidence_threshold=0.25,
        target_ids={0},
        use_masks=True,
        mask_min_area=0.0,
        mask_bottom_band_px=3,
    )

    assert selected['detected'] is True
    assert selected['mask_available'] == 1
    assert selected['selected_pixel_source'] == 'bbox_bottom'
    assert selected['selected_u'] == 20.0
    assert selected['selected_v'] == 60.0
    assert selected['bbox_bottom_u'] == 20.0
    assert selected['bbox_bottom_v'] == 60.0
    assert np.isclose(selected['mask_bottom_u'], 58.0 / 3.0)
    assert selected['mask_bottom_v'] == 60.0


def test_malformed_candidate_boxes_fail_closed_and_fall_through() -> None:
    result = _Result()
    result.boxes = _Boxes(
        cls=[0, 0, 0, 0],
        conf=[0.9, 0.85, 0.8, 0.6],
        xyxy=[
            [-1.0, 20.0, 30.0, 60.0],
            [10.0, 20.0, 10.0, 60.0],
            [10.0, 20.0, 30.0, float('nan')],
            [40.0, 50.0, 60.0, 90.0],
        ],
    )

    detections = collect_candidate_detections(result, {0})
    assert len(detections) == 1
    selected = select_best_detection(
        result,
        confidence_threshold=0.25,
        target_ids={0},
        use_masks=False,
        mask_min_area=0.0,
        mask_bottom_band_px=0.0,
    )

    assert selected['detected'] is True
    assert selected['selected_pixel_source'] == 'bbox_bottom'
    assert (selected['selected_u'], selected['selected_v']) == (50.0, 90.0)
