"""Validation and ordering for the coherent operational belief publication.

This module uses no ROS, cameras or ground truth. Freshness describes the
predicted state timestamp. A long camera outage does not invalidate a fresh
prediction whose motion support is explicitly valid.
"""
from dataclasses import dataclass
import json
import math

import numpy as np


OPERATIONAL_BELIEF_TOPIC = '/planner/belief_state'
# New delivery deadline for coherent predicted state. The initial value matches
# the existing actuator stop deadline; it was not historical mission policy and
# must be frozen explicitly for successor runs. It never limits anchor/camera age.
OPERATIONAL_BELIEF_TIMEOUT_S = 0.5


def _identity(data):
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('unsupported operational belief schema')
    epoch, revision = data.get('epoch'), data.get('revision')
    if not isinstance(epoch, str) or not epoch or epoch.strip() != epoch:
        raise ValueError('epoch must be a nonempty string')
    if type(revision) is not int or revision < 0:
        raise ValueError('revision must be a nonnegative integer')
    return epoch, revision


def _stamp(value, name, *, optional=False):
    if optional and value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be nonnegative integer nanoseconds')
    return value


@dataclass(frozen=True)
class OperationalBelief:
    epoch: str
    revision: int
    frame_id: str
    anchor_stamp_ns: int | None
    state_stamp_ns: int | None
    mean: tuple | None
    covariance: tuple | None
    valid: bool
    invalid_reason: str
    motion_supported: bool
    motion_support_json: str
    initialized: bool

    @property
    def identity(self):
        return self.epoch, self.state_stamp_ns, self.revision

    @property
    def motion_support(self):
        # Do not expose mutable data owned by the receive cache.
        return json.loads(self.motion_support_json)


def operational_belief_from_json(text):
    """Parse a complete publication, including an explicit invalid-state event."""
    try:
        data = json.loads(text)
        epoch, revision = _identity(data)
        valid, supported = data['valid'], data['motion_supported']
        if type(valid) is not bool or type(supported) is not bool:
            raise ValueError('valid and motion_supported must be booleans')
        frame, reason = data['frame_id'], data['invalid_reason']
        if not isinstance(frame, str) or not isinstance(reason, str):
            raise ValueError('frame_id and invalid_reason must be strings')
        anchor = _stamp(data['anchor_stamp_ns'], 'anchor_stamp_ns', optional=not valid)
        state = _stamp(data['state_stamp_ns'], 'state_stamp_ns', optional=not valid)
        raw_mean, raw_covariance = data['mean'], data['covariance']
        if raw_mean is None and not valid:
            mean = None
        else:
            if not isinstance(raw_mean, list) or len(raw_mean) != 3:
                raise ValueError('mean must contain x, y, yaw')
            mean = tuple(raw_mean)
        if raw_covariance is None and not valid:
            covariance = None
        else:
            if not isinstance(raw_covariance, list) or len(raw_covariance) != 3 or any(
                not isinstance(row, list) or len(row) != 3 for row in raw_covariance
            ):
                raise ValueError('covariance must be 3 by 3')
            covariance = tuple(tuple(row) for row in raw_covariance)
        support = json.dumps(data['motion_support'], allow_nan=False, sort_keys=True)
        initialized = data.get('initialized', revision > 0)
        if type(initialized) is not bool:
            raise ValueError('initialized must be boolean')
        return OperationalBelief(epoch, revision, frame, anchor, state, mean, covariance,
                                 valid, reason, supported, support, initialized)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'invalid operational belief: {exc}') from exc


def operational_belief_invalid_reason(belief, *, now_ns, expected_frame, max_age_s=OPERATIONAL_BELIEF_TIMEOUT_S):
    """Return an empty string only for a usable operational snapshot."""
    if belief is None:
        return 'missing_operational_belief'
    if not belief.valid:
        return belief.invalid_reason or 'producer_invalid_belief'
    if belief.invalid_reason:
        return 'contradictory_belief_validity'
    if not belief.initialized:
        return 'belief_not_initialized'
    if not belief.motion_supported:
        return 'unsupported_motion'
    if belief.frame_id != expected_frame or not belief.frame_id:
        return 'incompatible_belief_frame'
    if type(now_ns) is not int or now_ns < 0:
        return 'invalid_consumer_clock'
    if not math.isfinite(max_age_s) or max_age_s < 0:
        return 'invalid_operational_belief_timeout'
    if belief.state_stamp_ns is None or belief.anchor_stamp_ns is None:
        return 'missing_belief_timestamp'
    if belief.anchor_stamp_ns > belief.state_stamp_ns:
        return 'anchor_after_predicted_state'
    support = belief.motion_support
    if not isinstance(support, dict):
        return 'missing_motion_support'
    start, end = support.get('start_stamp_ns'), support.get('end_stamp_ns')
    source, gaps = support.get('source'), support.get('gaps')
    if (type(start) is not int or type(end) is not int or start < 0 or end < start
            or not isinstance(source, str) or not source or source.strip() != source
            or not isinstance(gaps, list)):
        return 'malformed_motion_support'
    if support.get('supported') is not True or gaps:
        return 'contradictory_motion_support'
    if start > belief.anchor_stamp_ns or end < belief.state_stamp_ns:
        return 'incomplete_motion_support'
    if source == 'none' and belief.state_stamp_ns > belief.anchor_stamp_ns:
        return 'missing_motion_source'
    age_ns = now_ns - belief.state_stamp_ns
    if age_ns < 0:
        return 'future_belief_state'
    if age_ns > round(max_age_s * 1_000_000_000):
        return 'stale_belief_state'
    try:
        entries = (*belief.mean, *(v for row in belief.covariance for v in row))
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in entries):
            return 'nonfinite_or_nonnumeric_belief'
        covariance = np.asarray(belief.covariance, dtype=float)
        if len(belief.mean) != 3 or covariance.shape != (3, 3):
            return 'malformed_belief_covariance'
        # Numerical symmetry/PSD tolerance only; never repair or inflate P.
        eps = 1e-12 * max(1., float(np.max(np.abs(covariance))))
        if np.max(np.abs(covariance - covariance.T)) > eps:
            return 'asymmetric_belief_covariance'
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.isfinite(eigenvalues).all():
            return 'invalid_belief_covariance_spectrum'
        if float(eigenvalues.min()) < -eps:
            return 'indefinite_belief_covariance'
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return 'malformed_belief_covariance'
    return ''


class OperationalBeliefReceiver:
    """Track epoch/revision order and expire a held operational snapshot.

    An older delivery cannot replace a newer event. A new invalid event clears
    usability. A duplicate cannot revive data invalidated by a malformed event.
    Clock rewind closes the previous producer epoch; only a new epoch can resume.
    """

    def __init__(self, *, expected_frame='map_bev', max_age_s=OPERATIONAL_BELIEF_TIMEOUT_S):
        if not isinstance(expected_frame, str) or not expected_frame:
            raise ValueError('expected_frame must be nonempty')
        if not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError('max_age_s must be finite and nonnegative')
        self.expected_frame, self.max_age_s = expected_frame, float(max_age_s)
        self.latest = None
        self.reason = 'missing_operational_belief'
        self._epoch, self._revision = None, -1
        self._closed_epochs = set()
        self._last_clock_ns = None
        self._last_payload = None
        self._last_state_stamp_ns = None
        self._last_event_key = None
        self._revision_anchor_stamp_ns = None

    def _clock(self, now_ns):
        if type(now_ns) is not int or now_ns < 0:
            self.latest = None
            self.reason = 'invalid_consumer_clock'
            return False
        if self._last_clock_ns is not None and now_ns < self._last_clock_ns:
            if self._epoch is not None:
                self._closed_epochs.add(self._epoch)
            self._epoch, self._revision = None, -1
            self.latest, self._last_payload, self._last_state_stamp_ns = None, None, None
            self._last_event_key = None
            self._revision_anchor_stamp_ns = None
            self.reason = 'clock_rewind_requires_new_belief_epoch'
        self._last_clock_ns = now_ns
        return True

    def receive_json(self, text, *, now_ns):
        if not self._clock(now_ns):
            return False
        try:
            data = json.loads(text)
            epoch, revision = _identity(data)
            if epoch in self._closed_epochs:
                return False
            if epoch == self._epoch and revision < self._revision:
                return False
            previous_revision = self._revision if epoch == self._epoch else -1
            if revision != previous_revision:
                self._revision_anchor_stamp_ns = None
            if epoch != self._epoch:
                if self._epoch is not None:
                    self._closed_epochs.add(self._epoch)
                self._last_state_stamp_ns = None
                self._last_event_key, self._last_payload = None, None
            # Keep a recognized newer revision even when the body is malformed:
            # a late older delivery cannot restore readiness after this event.
            self._epoch, self._revision = epoch, revision
            state_stamp = _stamp(data.get('state_stamp_ns'), 'state_stamp_ns', optional=not data.get('valid'))
            if (revision == previous_revision
                    and state_stamp is not None and self._last_state_stamp_ns is not None
                    and state_stamp < self._last_state_stamp_ns):
                return False
            payload = json.dumps(data, sort_keys=True, separators=(',', ':'))
            event_key = epoch, state_stamp, revision
            if event_key == self._last_event_key:
                if payload != self._last_payload:
                    self.latest = None
                    self.reason = 'conflicting_belief_revision'
                return False
            self._last_event_key, self._last_payload = event_key, payload
            belief = operational_belief_from_json(text)
            if belief.anchor_stamp_ns is not None:
                if (self._revision_anchor_stamp_ns is not None
                        and belief.anchor_stamp_ns != self._revision_anchor_stamp_ns):
                    self.latest = None
                    self.reason = 'conflicting_belief_anchor_revision'
                    return False
                self._revision_anchor_stamp_ns = belief.anchor_stamp_ns
            if (belief.state_stamp_ns is not None and self._last_state_stamp_ns is not None
                    and belief.state_stamp_ns < self._last_state_stamp_ns):
                self.latest = None
                self.reason = 'belief_state_time_regressed'
                return False
            if belief.state_stamp_ns is not None:
                self._last_state_stamp_ns = belief.state_stamp_ns
            self.latest = belief
            self.usable(now_ns=now_ns)
            return True
        except (TypeError, ValueError, OverflowError) as exc:
            self.latest = None
            self.reason = str(exc)
            return False

    def usable(self, *, now_ns):
        if not self._clock(now_ns):
            return None
        reason = operational_belief_invalid_reason(self.latest, now_ns=now_ns,
            expected_frame=self.expected_frame, max_age_s=self.max_age_s)
        if self.latest is not None:
            self.reason = reason
        return self.latest if not reason else None
