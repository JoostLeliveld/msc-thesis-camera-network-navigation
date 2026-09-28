"""Runtime extraction of the commissioned 16x16 robot-visibility matrix."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from PIL import Image as PilImage
from torch.nn import functional as F


GRID_SIZE = 16
CONTEXT_SIZE = 96
BLUE_DOMINANCE_CENTER = 0.035
BLUE_DOMINANCE_SCALE = 0.035
CHROMA_CENTER = 0.075
CHROMA_SCALE = 0.040
BLUE_VALUE_CENTER = 0.12
BLUE_VALUE_SCALE = 0.055


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def soft_target_blue(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB input must have shape [height,width,3]")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("floating RGB input must be finite and in [0,1]")
    red, green, blue = (image[..., index] for index in range(3))
    dominance = blue - np.maximum(red, green)
    chroma = np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))
    return np.asarray(
        _sigmoid((dominance - BLUE_DOMINANCE_CENTER) / BLUE_DOMINANCE_SCALE)
        * _sigmoid((chroma - CHROMA_CENTER) / CHROMA_SCALE)
        * _sigmoid((blue - BLUE_VALUE_CENTER) / BLUE_VALUE_SCALE),
        dtype=np.float32,
    )


def context_bounds(bbox_xyxy: Sequence[float], image_shape: Sequence[int]) -> tuple[int, int, int, int]:
    bbox = np.asarray(bbox_xyxy, dtype=float)
    height, width = map(int, image_shape)
    x0, y0, x1, y1 = bbox
    box_width, box_height = x1 - x0, y1 - y0
    left = int(max(0.0, x0 - 0.5 * box_width))
    top = int(max(0.0, y0 - 0.5 * box_height))
    right = int(min(float(width), x1 + 0.5 * box_width))
    bottom = int(min(float(height), y1 + 0.5 * box_height))
    if right <= left or bottom <= top:
        raise ValueError("context crop is empty")
    return left, top, right, bottom


def visibility_grid_from_saved_context(
    context_chw: np.ndarray, bbox_xyxy: Sequence[float], image_shape: Sequence[int]
) -> np.ndarray:
    crop = np.asarray(context_chw)
    if crop.shape != (3, CONTEXT_SIZE, CONTEXT_SIZE) or crop.dtype != np.uint8:
        raise ValueError("saved context crop must be uint8 with shape [3,96,96]")
    left, top, right, bottom = context_bounds(bbox_xyxy, image_shape)
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    scale_x = CONTEXT_SIZE / float(right - left)
    scale_y = CONTEXT_SIZE / float(bottom - top)
    bx0 = int(np.clip(math.floor((x0 - left) * scale_x), 0, CONTEXT_SIZE - 1))
    by0 = int(np.clip(math.floor((y0 - top) * scale_y), 0, CONTEXT_SIZE - 1))
    bx1 = int(np.clip(math.ceil((x1 - left) * scale_x), bx0 + 1, CONTEXT_SIZE))
    by1 = int(np.clip(math.ceil((y1 - top) * scale_y), by0 + 1, CONTEXT_SIZE))
    score = soft_target_blue(crop.transpose(1, 2, 0))[by0:by1, bx0:bx1]
    return np.asarray(
        F.adaptive_avg_pool2d(torch.from_numpy(score)[None, None], (GRID_SIZE, GRID_SIZE))[0].numpy(),
        dtype=np.float32,
    )


def visibility_grid_from_bgr_frame(image_bgr: np.ndarray, bbox_xyxy: Sequence[float]) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("BGR frame must be uint8 with shape [height,width,3]")
    bounds = context_bounds(bbox_xyxy, image.shape[:2])
    rgb = image[..., ::-1]
    context = PilImage.fromarray(rgb).crop(bounds).resize(
        (CONTEXT_SIZE, CONTEXT_SIZE), PilImage.Resampling.BILINEAR
    )
    return visibility_grid_from_saved_context(
        np.asarray(context, dtype=np.uint8).transpose(2, 0, 1), bbox_xyxy, image.shape[:2]
    )
