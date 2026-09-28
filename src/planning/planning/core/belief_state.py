"""Owned immutable recursive state and prediction provenance (no ROS dependency)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def checked_state(mean, covariance):
    """Validate a proposed state before it can replace the last valid estimate."""
    m = np.array(mean, dtype=float, copy=True)
    P = np.array(covariance, dtype=float, copy=True)
    if m.shape != (3,) or P.shape != (3, 3):
        raise ValueError('belief needs a three-dimensional mean and full 3x3 covariance')
    if not np.isfinite(m).all() or not np.isfinite(P).all():
        raise ValueError('nonfinite belief mean or covariance')
    if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
        raise ValueError('asymmetric belief covariance')
    P = (P + P.T) * .5
    if float(np.linalg.eigvalsh(P).min()) < -1e-10:
        raise ValueError('indefinite belief covariance')
    return m, P


@dataclass(frozen=True)
class MotionSupport:
    start_ns: int
    end_ns: int
    source: str = 'none'
    gaps: tuple = ()  # (start_ns, end_ns, reason)

    def __post_init__(self):
        if (type(self.start_ns) is not int or type(self.end_ns) is not int
                or self.start_ns < 0 or self.end_ns < self.start_ns):
            raise ValueError('motion support requires ordered integer times')
        if not isinstance(self.source, str) or not self.source:
            raise ValueError('motion support requires an explicit source')
        for a,b,reason in self.gaps:
            if (type(a) is not int or type(b) is not int
                    or a < self.start_ns or b > self.end_ns or b <= a or not reason):
                raise ValueError('motion support gap is outside its interval')

    @property
    def supported(self):
        return self.end_ns >= self.start_ns and not self.gaps

    def to_dict(self):
        return dict(start_stamp_ns=self.start_ns, end_stamp_ns=self.end_ns,
                    source=self.source, supported=self.supported,
                    gaps=[dict(start_stamp_ns=a, end_stamp_ns=b, reason=c)
                          for a, b, c in self.gaps])

    @classmethod
    def from_dict(cls, value):
        return cls(int(value['start_stamp_ns']), int(value['end_stamp_ns']), str(value['source']),
                   tuple((int(g['start_stamp_ns']), int(g['end_stamp_ns']), str(g['reason']))
                         for g in value.get('gaps', ())))

    def following(self, previous):
        """An unsupported historical interval is not repaired by a newer clock."""
        if previous is None or previous.supported:
            return self
        return MotionSupport(min(previous.start_ns, self.start_ns), self.end_ns,
                             self.source, tuple(dict.fromkeys(previous.gaps + self.gaps)))


@dataclass(frozen=True)
class BeliefRecord:
    mean: tuple
    covariance: tuple
    stamp_ns: int
    frame_id: str
    epoch: str
    revision: int
    motion_support: MotionSupport

    @classmethod
    def create(cls, mean, covariance, stamp_ns, frame_id, epoch, revision,
               motion_support=None):
        m, P = checked_state(mean, covariance)
        if type(stamp_ns) is not int or stamp_ns < 0:
            raise ValueError('belief time must be nonnegative integer nanoseconds')
        if not frame_id or not epoch:
            raise ValueError('belief frame and epoch must be explicit')
        support = motion_support or MotionSupport(stamp_ns, stamp_ns)
        if support.end_ns != stamp_ns or support.start_ns > stamp_ns:
            raise ValueError('belief motion support must end at its state time')
        if type(revision) is not int or revision < 0:
            raise ValueError('belief revision must be a nonnegative integer')
        return cls(tuple(float(v) for v in m), tuple(tuple(float(v) for v in row) for row in P),
                   stamp_ns, str(frame_id), str(epoch), int(revision), support)

    def arrays(self):
        return np.array(self.mean), np.array(self.covariance)


@dataclass(frozen=True)
class PredictionSnapshot:
    anchor: BeliefRecord
    mean: tuple
    covariance: tuple
    state_stamp_ns: int
    motion_support: MotionSupport
    valid: bool = True
    invalid_reason: str = ''
    goal_revision: int = 0

    @classmethod
    def create(cls, anchor, mean, covariance, state_stamp_ns, motion_support,
               *, valid=True, invalid_reason='', goal_revision=0):
        m, P = checked_state(mean, covariance)
        if (type(state_stamp_ns) is not int or state_stamp_ns < anchor.stamp_ns
                or motion_support.start_ns > anchor.stamp_ns
                or motion_support.end_ns != state_stamp_ns):
            raise ValueError('prediction time and motion support disagree')
        if valid and not motion_support.supported:
            raise ValueError('unsupported prediction cannot be declared valid')
        return cls(anchor, tuple(m), tuple(tuple(row) for row in P), int(state_stamp_ns),
                   motion_support, bool(valid), str(invalid_reason), int(goal_revision))

    def to_dict(self):
        return dict(schema_version=1, initialized=True, epoch=self.anchor.epoch, revision=self.anchor.revision,
                    frame_id=self.anchor.frame_id, anchor_stamp_ns=self.anchor.stamp_ns,
                    state_stamp_ns=self.state_stamp_ns, mean=list(self.mean),
                    covariance=[list(row) for row in self.covariance], valid=self.valid,
                    invalid_reason=self.invalid_reason,
                    motion_supported=self.motion_support.supported,
                    motion_support=self.motion_support.to_dict())

    def planner_meta(self):
        return dict(belief_epoch=self.anchor.epoch, belief_revision=self.anchor.revision,
                    belief_stamp_ns=self.anchor.stamp_ns, prediction_stamp_ns=self.state_stamp_ns,
                    belief_frame_id=self.anchor.frame_id, belief_valid=self.valid,
                    motion_supported=self.motion_support.supported,
                    motion_support=self.motion_support.to_dict(), goal_revision=self.goal_revision,
                    invalid_reason=self.invalid_reason)
