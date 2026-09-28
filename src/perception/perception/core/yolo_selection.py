from __future__ import annotations

import math

import numpy as np


def target_class_ids(names, class_name: str, class_id: int):
    if int(class_id) >= 0:
        return {int(class_id)}
    target = str(class_name or '').strip().lower()
    if target in ('', '*', 'any', 'all'):
        return None
    if isinstance(names, dict):
        return {int(idx) for idx, name in names.items() if str(name).strip().lower() == target}
    return set()


def mask_bottom(points: np.ndarray, band_px: float) -> tuple[float, float]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return math.nan, math.nan
    v_bottom = float(np.max(points[:, 1]))
    band = points[points[:, 1] >= v_bottom - max(float(band_px), 0.0)]
    if band.size == 0:
        return float(np.mean(points[:, 0])), v_bottom
    return float(np.mean(band[:, 0])), v_bottom


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return math.nan
    import cv2
    return float(cv2.contourArea(points[:, :2]))


def collect_candidate_detections(result, target_ids):
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    # Box-geometry validity moved here from an `assert` on the selected detection further
    # down.  An assert in a runtime detector node turns a malformed box into a dead node, and
    # it only ever guarded the top-scoring candidate.  BEHAVIOUR NOTE for anyone comparing
    # detection rates across this change: a frame whose candidates are all non-finite,
    # negative-origin or zero-area now reports `detected=False` where it previously raised,
    # and a frame whose BEST candidate was malformed now falls through to the runner-up
    # instead of raising.  Every well-formed box is selected exactly as before, so campaign
    # logs recorded on valid detections are unaffected.
    valid = np.isfinite(conf)
    valid &= np.all(np.isfinite(xyxy[:, :4]), axis=1)
    valid &= xyxy[:, 0] >= 0.0
    valid &= xyxy[:, 1] >= 0.0
    valid &= xyxy[:, 2] > xyxy[:, 0]
    valid &= xyxy[:, 3] > xyxy[:, 1]
    if target_ids is not None:
        valid &= np.asarray([int(c) in target_ids for c in cls], dtype=bool)
    idxs = np.flatnonzero(valid)
    if idxs.size == 0:
        return []

    masks = getattr(result, 'masks', None)
    mask_polygons = getattr(masks, 'xy', None) if masks is not None else None
    detections = []
    for idx in idxs:
        idx = int(idx)
        polygon = None
        if mask_polygons is not None and idx < len(mask_polygons):
            pts = np.asarray(mask_polygons[idx], dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                pts = pts[:, :2]
                pts = pts[np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])]
                if pts.shape[0] >= 3:
                    polygon = pts
        detections.append({
            'bbox': xyxy[idx, :4].astype(float),
            'confidence': float(conf[idx]),
            'class_id': int(cls[idx]),
            'polygon': polygon,
        })
    detections.sort(key=lambda item: float(item['confidence']), reverse=True)
    return detections


def select_best_detection(
    result,
    *,
    target_ids,
    confidence_threshold: float,
    use_masks: bool,
    mask_min_area: float,
    mask_bottom_band_px: float,
):
    detections = collect_candidate_detections(result, target_ids)
    if not detections:
        return {
            'detected': False,
            'detected_after_threshold': False,
            'raw_best_score': 0.0,
            'selected_score': 0.0,
            'confidence': 0.0,
            'best_class_id': math.nan,
            'class_id': math.nan,
            'bbox_xyxy': None,
            'mask_area': math.nan,
            'mask_bottom_u': math.nan,
            'mask_bottom_v': math.nan,
            'mask_centroid_u': math.nan,
            'mask_centroid_v': math.nan,
            'bbox_bottom_u': math.nan,
            'bbox_bottom_v': math.nan,
            'bbox_center_u': math.nan,
            'bbox_center_v': math.nan,
            'selected_u': math.nan,
            'selected_v': math.nan,
            'selected_pixel_source': 'none',
            'mask_available': 0,
            'n_candidates': 0,
            'detection': None,
            'candidates': detections,
        }

    detection = detections[0]
    bbox = np.asarray(detection['bbox'], dtype=float)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    polygon = detection['polygon']
    mask_area = math.nan
    mask_available = 0
    bbox_center_u = float(0.5 * (x0 + x1))
    bbox_center_v = float(0.5 * (y0 + y1))
    bbox_bottom_u = bbox_center_u
    bbox_bottom_v = float(y1)
    mask_bottom_u = math.nan
    mask_bottom_v = math.nan
    mask_centroid_u = math.nan
    mask_centroid_v = math.nan

    # The deployed state estimator inverts the bbox bottom-centre through its configured
    # contact plane. Alternative statistics remain experiment-local until their complete
    # observation contract and sequential-use semantics are approved together.
    selected_source = 'bbox_bottom'
    selected_u = bbox_bottom_u
    selected_v = bbox_bottom_v
    if use_masks and polygon is not None:
        area = polygon_area(polygon)
        if math.isfinite(area) and area >= float(mask_min_area):
            mask_area = float(area)
            mask_available = 1
            mask_centroid_u = float(np.mean(polygon[:, 0]))
            mask_centroid_v = float(np.mean(polygon[:, 1]))
            mask_bottom_u, mask_bottom_v = mask_bottom(polygon, float(mask_bottom_band_px))
    raw_best_score = float(detection['confidence'])
    detected_after_threshold = bool(raw_best_score >= float(confidence_threshold))

    return {
        'detected': bool(detected_after_threshold),
        'detected_after_threshold': bool(detected_after_threshold),
        'raw_best_score': raw_best_score,
        'selected_score': raw_best_score,
        'confidence': raw_best_score,
        'best_class_id': int(detection['class_id']),
        'class_id': int(detection['class_id']),
        'bbox_xyxy': bbox,
        'mask_area': float(mask_area) if math.isfinite(mask_area) else math.nan,
        'mask_bottom_u': float(mask_bottom_u) if math.isfinite(mask_bottom_u) else math.nan,
        'mask_bottom_v': float(mask_bottom_v) if math.isfinite(mask_bottom_v) else math.nan,
        'mask_centroid_u': float(mask_centroid_u) if math.isfinite(mask_centroid_u) else math.nan,
        'mask_centroid_v': float(mask_centroid_v) if math.isfinite(mask_centroid_v) else math.nan,
        'bbox_bottom_u': float(bbox_bottom_u),
        'bbox_bottom_v': float(bbox_bottom_v),
        'bbox_center_u': float(bbox_center_u),
        'bbox_center_v': float(bbox_center_v),
        'selected_u': float(selected_u),
        'selected_v': float(selected_v),
        'selected_pixel_source': selected_source,
        'mask_available': int(mask_available),
        'n_candidates': int(len(detections)),
        'detection': detection,
        'candidates': detections,
    }
