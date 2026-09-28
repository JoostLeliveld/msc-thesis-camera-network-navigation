"""Coverage checks for timestamped zero-order-held motion inputs.

A long camera interval can still have dense measured odometry. Its duration is
not a missing-motion interval. These checks use only the recorded input clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from planning.core.belief_state import MotionSupport


def covers_interval(entries, start_s: float, end_s: float, max_gap_s: float) -> bool:
    """Require a finite, ordered input at/before start and bounded gaps to end.

    Future samples cannot fill a past gap. A stale sample before the interval
    also fails, even if the interval itself is short. This checks temporal
    support, not whether the odometry noise model is statistically calibrated.
    """
    if not all(math.isfinite(v) for v in (start_s, end_s, max_gap_s)):
        return False
    if end_s <= start_s or max_gap_s <= 0:
        return False
    relevant = []
    previous = None
    last_stamp = -math.inf
    for entry in entries:
        t, v, w = entry
        if not all(math.isfinite(x) for x in (t, v, w)) or t < last_stamp:
            return False
        last_stamp = t
        if t <= start_s:
            previous = t
        elif t <= end_s:
            relevant.append(t)
    if previous is None or start_s-previous > max_gap_s + 1e-9:
        return False
    for t in [*relevant, end_s]:
        if t-previous > max_gap_s + 1e-9:
            return False
        previous = t
    return True


@dataclass(frozen=True)
class MotionHistorySnapshot:
    """One immutable view of both motion buffers plus the selected source.

    Coverage is checked against the entries that will actually be replayed. When
    those are two separate reads of live state, a trim or append in between makes
    the check describe a history the prediction never sees. Both lists are frozen
    because prediction may fall back from odometry to commands: freezing only the
    preferred list would leave that fallback reading mutable state.

    This carries temporal support, not a claim that the motion is calibrated.
    """

    odom: tuple
    cmd: tuple
    use_odom: bool
    headings: tuple = ()  # accepted (integer nanoseconds, odometry yaw)
    gaps: tuple = ()  # known omitted odometry intervals (integer ns, ns, reason)

    @classmethod
    def capture(cls, odom_entries, cmd_entries, use_odom: bool, headings=(), gaps=()) -> "MotionHistorySnapshot":
        """Freeze both buffers into tuples of primitive floats.

        Must be called while the caller holds the lock that guards the buffers.
        """
        return cls(
            odom=tuple((float(t), float(v), float(w)) for t, v, w in odom_entries),
            cmd=tuple((float(t), float(v), float(w)) for t, v, w in cmd_entries),
            use_odom=bool(use_odom),
            headings=tuple((int(t), float(yaw)) for t, yaw in headings),
            gaps=tuple((int(a), int(b), str(reason)) for a,b,reason in gaps),
        )

    @property
    def selected(self) -> tuple:
        """The entries the configured source would replay."""
        return self.odom if self.use_odom else self.cmd


@dataclass(frozen=True)
class ReplayPlan:
    start_ns: int
    end_ns: int
    segments: tuple  # (start_ns, end_ns, v, omega)
    support: MotionSupport
    event_count: int = 0


def plan_replay(snapshot: MotionHistorySnapshot, start_ns: int, end_ns: int,
                max_gap_s: float) -> ReplayPlan:
    """Partition actual elapsed time once; annotate every unsupported interval.

    Missing prefix/empty history retains the zero-motion assumption. A stale
    held velocity remains a stated fallback, never certified measured motion.
    No sensor gate or process-noise value is selected here.
    """
    start_ns, end_ns = int(start_ns), int(end_ns)
    if end_ns < start_ns:
        raise ValueError('motion replay target precedes its anchor')
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('motion support needs a positive finite gap bound')
    gap_ns = round(max_gap_s * 1e9)

    def normalized(entries):
        result = []
        last = -1
        for t, v, w in entries:
            if not all(math.isfinite(x) for x in (t, v, w)):
                raise ValueError('nonfinite motion input')
            ns = round(t * 1e9)
            if ns < 0 or ns < last:
                raise ValueError('unordered or negative motion timestamp')
            if result and ns == last:
                result[-1] = (ns, v, w)  # command receipts may share a clock tick
            else:
                result.append((ns, v, w))
            last = ns
        return result

    entries = normalized(snapshot.selected)
    source = 'odom' if snapshot.use_odom else 'command'
    if snapshot.use_odom and not any(t <= end_ns for t, _, _ in entries):
        entries, source = normalized(snapshot.cmd), 'command'
    if not any(t <= end_ns for t, _, _ in entries):
        source = 'none'
    previous = next((row for row in reversed(entries) if row[0] <= start_ns), None)
    relevant = [row for row in entries if start_ns < row[0] <= end_ns]
    segments, gaps = [], []
    cursor = start_ns
    for boundary in [*relevant, (end_ns, 0., 0.)]:
        end = boundary[0]
        if end > cursor:
            v, w = (previous[1], previous[2]) if previous is not None else (0., 0.)
            segments.append((cursor, end, v, w))
            if previous is None:
                gaps.append((cursor, end, 'missing_prefix' if source != 'none' else 'missing_motion'))
            elif end > previous[0] + gap_ns:
                gaps.append((max(cursor, previous[0] + gap_ns), end, 'stale_motion'))
        previous, cursor = boundary, end
    if source == 'odom':
        gaps.extend((max(a,start_ns), min(b,end_ns), reason) for a,b,reason in snapshot.gaps
                    if min(b,end_ns) > max(a,start_ns))
    return ReplayPlan(start_ns, end_ns, tuple(segments),
                      MotionSupport(start_ns, end_ns, source, tuple(gaps)), len(relevant))
