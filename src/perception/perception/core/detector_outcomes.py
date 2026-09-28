"""Versioned detector identity and durable outcome records, independent of ROS.

source_batch_id remains a logical all-camera cycle. Each native predict call is
an invocation; each received image has a separate exact-stamp/content identity.
Neither identifier claims a hardware capture sequence absent from sensor_msgs/Image.
"""
from __future__ import annotations

import hashlib

# Public compatibility exports; reliability/experiments import the common module
# directly so no dependency from reliability back to perception is introduced.
from unav_common.camera_outcomes import (
    OutcomeJournal, journal_path, read_journal, canonical_json,
    OUTCOME_SCHEMA, OUTCOME_HISTORY_DEPTH, DEFAULT_JOURNAL_MAX_BYTES,
)


def image_content_sha256(*, encoding, height, width, step, data):
    metadata = canonical_json(dict(encoding=str(encoding).lower(), height=int(height),
                                   width=int(width), step=int(step))).encode('utf-8')
    digest = hashlib.sha256(metadata + b'\n')
    try:
        raw = memoryview(data)
    except TypeError:
        raw = memoryview(bytes(data))
    digest.update(raw)
    return digest.hexdigest()


def source_frame_id(epoch, camera_id, stamp_ns, content_sha256):
    if not epoch or not camera_id or isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
        raise ValueError('frame identity requires epoch, camera and exact nonnegative stamp')
    if len(content_sha256) != 64 or any(c not in '0123456789abcdef' for c in content_sha256):
        raise ValueError('frame identity requires a SHA-256 image digest')
    return f'frame:{epoch}:{camera_id}:{stamp_ns}:{content_sha256}'


def frame_member(frame):
    return dict(camera_id=frame.camera_id, capture_stamp_ns=frame.stamp_ns,
                source_frame_id=frame.source_frame_id,
                content_sha256=frame.content_sha256,
                image_receive_stamp_s=frame.receive_stamp_s,
                image_receive_wall_s=frame.receive_wall_s)
