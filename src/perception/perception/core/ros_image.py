"""Small ROS image conversion helpers that avoid the NumPy/cv_bridge ABI boundary."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


def image_msg_to_bgr8(msg: Image) -> np.ndarray:
    """Convert common 8-bit ROS image encodings to a contiguous BGR image.

    This intentionally avoids ``cv_bridge`` because ROS Humble's binary
    ``cv_bridge`` can fail to import when the active Python environment has a
    NumPy 2.x wheel while the extension was compiled against NumPy 1.x.
    """
    encoding = str(msg.encoding or '').strip().lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0:
        raise ValueError(f'Invalid image shape height={height}, width={width}')

    channel_count = {
        'bgr8': 3,
        'rgb8': 3,
        'bgra8': 4,
        'rgba8': 4,
        'mono8': 1,
        '8uc1': 1,
    }.get(encoding)
    if channel_count is None:
        raise ValueError(f'Unsupported image encoding {msg.encoding!r}; expected bgr8/rgb8/bgra8/rgba8/mono8')

    min_step = width * channel_count
    if step < min_step:
        raise ValueError(f'Image step {step} is too small for {width}x{channel_count} encoding {encoding}')

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError(f'Image data has {raw.size} bytes, expected at least {expected}')

    rows = raw[:expected].reshape(height, step)
    pixels = rows[:, :min_step].reshape(height, width, channel_count)

    if encoding == 'bgr8':
        bgr = pixels
    elif encoding == 'rgb8':
        bgr = pixels[..., ::-1]
    elif encoding == 'bgra8':
        bgr = pixels[..., :3]
    elif encoding == 'rgba8':
        bgr = pixels[..., [2, 1, 0]]
    else:
        bgr = np.repeat(pixels, 3, axis=2)

    return np.ascontiguousarray(bgr)
