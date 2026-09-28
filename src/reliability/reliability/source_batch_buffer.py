"""Bounded, ROS-free receipt transactions for all-camera detector cycles.

Time orders evidence within one process epoch; it is not a replacement for the
source ID. Clock reset requires a coordinated runtime restart. Per-camera high
water marks prevent replay even after bounded closed-ID metadata is discarded.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, is_dataclass
import json
import math


class SourceBatchBuffer:
    def __init__(self, camera_ids, *, timeout_s=2.0, capacity=64, on_event=None):
        self.camera_ids = tuple(camera_ids)
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError('camera_ids must be nonempty and unique')
        if not math.isfinite(timeout_s) or timeout_s <= 0 or capacity < 1:
            raise ValueError('batch timeout and capacity must be positive')
        self.timeout_s, self.capacity = float(timeout_s), int(capacity)
        self.pending = OrderedDict()
        self.closed = OrderedDict()
        self.last_received = {cid: -math.inf for cid in self.camera_ids}
        self.completed_stamp = -math.inf
        self.on_event = on_event or (lambda event: None)

    def _close(self, bid, reason, now):
        first, observations = self.pending.pop(bid)
        self.closed[bid] = None
        while len(self.closed) > self.capacity * 4:
            self.closed.popitem(last=False)
        self.on_event(dict(source_batch_id=bid, status=reason,
                           received_camera_ids=list(observations),
                           missing_camera_ids=[c for c in self.camera_ids if c not in observations],
                           pending_wall_s=max(0.0, now - first)))
        return observations

    def expire(self, now):
        if not math.isfinite(now):
            raise ValueError('receipt wall time must be finite')
        for bid, (first, _) in list(self.pending.items()):
            if now - first > self.timeout_s:
                self._close(bid, 'incomplete_timeout', now)

    def offer(self, observation, now):
        self.expire(now)
        bid, cid = observation.source_batch_id, observation.camera_id
        stamp = float(observation.timestamp_s)
        if not bid or cid not in self.last_received or not math.isfinite(stamp):
            raise ValueError('invalid source batch member')
        if bid in self.closed:
            return None
        existing = self.pending.get(bid)
        if existing is not None and cid in existing[1]:
            # Identical repeated delivery is harmless; conflicting payloads abort
            # the entire pending transaction rather than choose by arrival order.
            original = existing[1][cid]
            def encoded(item):
                fields = asdict(item) if is_dataclass(item) else vars(item)
                return json.dumps(fields, sort_keys=True, allow_nan=True)
            if encoded(original) != encoded(observation):
                self._close(bid, 'conflicting_duplicate', now)
            return None
        if stamp <= self.last_received[cid]:
            self.on_event(dict(source_batch_id=bid, status='member_rejected_nonincreasing_stamp',
                               camera_id=cid, capture_stamp_s=stamp,
                               last_received_capture_stamp_s=self.last_received[cid]))
            return None
        self.last_received[cid] = stamp
        if bid not in self.pending:
            if len(self.pending) >= self.capacity:
                self._close(next(iter(self.pending)), 'incomplete_capacity', now)
            self.pending[bid] = (now, {})
        self.pending[bid][1][cid] = observation
        if len(self.pending[bid][1]) != len(self.camera_ids):
            return None
        result = self._close(bid, 'complete', now)
        self.completed_stamp = max(o.timestamp_s for o in result.values())
        # Preserve latest-complete scheduling, but account for every displaced set.
        for older in list(self.pending):
            self._close(older, 'superseded_by_complete_batch', now)
        return {cid: result[cid] for cid in self.camera_ids}
