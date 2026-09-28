"""Pure checks of observation meaning, separate from statistical admission.

Pixels are original-image coordinates. Bbox maxima are continuous half-open
edges and may equal width/height; no clipping or correction is performed here.
"""
import math
import numpy as np


def _finite_vector(value, length, name):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must contain finite numbers') from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must have {length} finite entries')
    return array


def validate_pixel_covariance(covariance):
    """Validate full pixel covariance without symmetrizing or regularizing it."""
    R = np.asarray(covariance, dtype=float)
    if R.shape != (2, 2) or not np.isfinite(R).all():
        raise ValueError('pixel covariance must be finite 2x2')
    if not np.allclose(R, R.T, atol=1e-12, rtol=0):
        raise ValueError('pixel covariance must be symmetric')
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError('pixel covariance must be positive definite') from exc
    return R


def validate_observation_geometry(observation, camera, *, expected_camera_id,
                                  expected_calibration_id, expected_image_frame_id,
                                  require_bbox=True, require_image_dimensions=False):
    """Raise ValueError before projection for a semantically incompatible reading.

    The caller supplies authoritative identities; they are never inferred from
    the incoming message. This function does not decide freshness, quorum or NIS.
    Legacy messages may omit original dimensions unless strict dimension mode
    is selected. Supplied dimensions must match the configured camera exactly.
    Set require_image_dimensions=True for producers carrying original size.
    ``require_bbox=True`` selects the bbox-bottom observation contract. Other
    callers may use an explicitly declared mask-bottom source with it false.
    """
    from reliability.contracts import SCHEMA_VERSION
    if observation.schema_version != SCHEMA_VERSION:
        raise ValueError('unsupported observation schema')
    for field, expected in [('camera_id', expected_camera_id),
                            ('calibration_id', expected_calibration_id),
                            ('image_frame_id', expected_image_frame_id)]:
        if not isinstance(expected, str) or not expected:
            raise ValueError(f'expected {field} must be explicitly configured')
        if getattr(observation, field, None) != expected:
            raise ValueError(f'observation {field} differs from configured identity')
    stamp = float(observation.timestamp_s)
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError('capture timestamp must be finite and nonnegative')
    width, height = float(camera.img_width), float(camera.img_height)
    if not all(math.isfinite(x) and x > 0 and x.is_integer() for x in (width, height)):
        raise ValueError('camera image dimensions must be positive integers')
    for field, expected in [('image_width', width), ('image_height', height),
                            ('image_width_px', width), ('image_height_px', height)]:
        value = getattr(observation, field, None)
        if value is not None and float(value) != expected:
            raise ValueError(f'{field} differs from camera calibration')
    if require_image_dimensions:
        for axis in ('width', 'height'):
            if all(getattr(observation, name, None) is None for name in
                   (f'image_{axis}', f'image_{axis}_px')):
                raise ValueError(f'original image {axis} is required')
    validate_pixel_covariance(observation.conditional_cov_uv)
    if not observation.detection_valid:
        return
    pixel = _finite_vector(observation.pixel_uv, 2, 'selected pixel')
    if not (0 <= pixel[0] <= width and 0 <= pixel[1] <= height):
        raise ValueError('selected pixel outside original image')
    box = getattr(observation, 'bbox_xyxy', None)
    bottom = None
    if box is not None:
        x0, y0, x1, y1 = _finite_vector(box, 4, 'bbox')
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('bbox must have positive half-open extents inside original image')
        bottom = np.array([(x0+x1)/2, y1])
        reported = getattr(observation, 'bbox_bottom_uv', None)
        if reported is not None and not np.allclose(
                _finite_vector(reported, 2, 'bbox bottom'), bottom, atol=1e-9, rtol=0):
            raise ValueError('bbox bottom differs from bbox edges')
    source = observation.selected_pixel_source
    if require_bbox and (bottom is None or source != 'bbox_bottom'):
        raise ValueError('bbox observation requires bbox_bottom source and bbox')
    if source == 'bbox_bottom':
        expected = bottom
    elif source == 'mask_bottom' and observation.mask_available:
        expected = _finite_vector(observation.mask_bottom_uv, 2, 'mask bottom')
    else:
        raise ValueError('valid detection requires a supported selected pixel source')
    if expected is None or not np.allclose(pixel, expected, atol=1e-9, rtol=0):
        raise ValueError('selected pixel differs from its declared source')
