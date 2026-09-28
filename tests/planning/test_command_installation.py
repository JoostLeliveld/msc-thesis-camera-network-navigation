"""An ordinary safe stop must cancel planning work that began before it.

Clearing the active tape was not enough. A solve already in flight installed its
result immediately afterwards, so a commanded stop lasted only until the solver
returned. These regressions drive the two installation routes -- the hierarchical
LOCAL installer and the direct solver's ``_after_plan_result`` -- through
event-controlled interleavings with fake publishers. No sleeps, no live ROS.

They also cover the computation-age gate, which was assigned on the LOCAL path but
never read there, and never assigned on the direct path.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from planning.nodes.efe_agent_node import EfeAgentNode
from test_planner_node_correction_wiring import _Clock
from test_runtime_transactions import Publisher, command_node


def _install_node(*, dt=.25):
    """A node ready to take an installed tape, with nothing installed yet."""
    node = command_node()
    node.dt = dt
    node._active_controls = None
    node._active_plan_started_at = None
    node._active_controls_original_len = 0
    node._command_stop_generation = 0
    node._active_plan_request = None
    node._fatal_stop_triggered = False
    node.get_logger = lambda: _SilentLogger()
    return node


class _SilentLogger:
    def warn(self, *_a, **_k):
        pass

    def info(self, *_a, **_k):
        pass


class _Result:
    """The minimum a direct solver result needs to reach installation."""

    def __init__(self, controls):
        self.controls = np.asarray(controls, dtype=float)
        self.rollout_valid = True
        self.invalid_reason = ''
        self.min_predicted_obstacle_distance_m = 1.0
        self.terminal_goal_distance_pred = 0.0


def _nonzero(messages):
    return [m for m in messages if m.linear.x != 0. or m.angular.z != 0.]


# --- A. ordinary-stop cancellation -----------------------------------------

def test_a_stop_during_computation_cancels_the_direct_solver_result():
    node = _install_node()
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None

    request = node._capture_plan_request()
    node._active_plan_request = request
    # The stop lands while the solver is still working.
    node._publish_safe_stop_command()
    node.cmd_pub.messages.clear()

    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert node._active_controls is None, 'a cancelled result installed a tape'
    assert not _nonzero(node.cmd_pub.messages)


def test_a_cancelled_result_does_not_erase_a_newer_replacement_tape():
    """The discard must be total: no tape clearing, no extra stop.

    A cancelled result that still issued its rejection side effects would wipe a
    tape a newer request had already installed.
    """
    node = _install_node()
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None

    stale_request = node._capture_plan_request()
    node._publish_safe_stop_command()

    # A fresh request, captured AFTER the stop, installs its own tape.
    replacement = np.array([[.1, -.2]])
    node._active_controls = replacement
    node._active_plan_started_at = _Clock(10.).now()
    node.cmd_pub.messages.clear()

    # Now the old cancelled solve returns.
    node._active_plan_request = stale_request
    node._after_plan_result(_Result([[.9, .9], [.9, .9]]))

    assert node._active_controls is replacement, 'the replacement tape was erased'
    assert not node.cmd_pub.messages


def test_a_request_captured_after_an_ordinary_stop_may_resume():
    node = _install_node()
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None

    node._publish_safe_stop_command()
    node.cmd_pub.messages.clear()

    node._active_plan_request = node._capture_plan_request()
    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert node._active_controls is not None
    assert _nonzero(node.cmd_pub.messages), 'an ordinary stop must not latch'


def test_rejected_local_replacement_preserves_the_admitted_active_tape():
    node = _install_node()
    node.use_hierarchical = True
    node._hier_phase = 'LOCAL'
    active = np.array([[.25, .1], [.25, .0]])
    node._active_controls = active
    node._active_plan_started_at = node.get_clock().now()
    node._active_controls_original_len = len(active)
    request = node._capture_plan_request()
    node._active_plan_request = request

    node._reject_current_request(request, 'replacement failed validation')

    assert node._active_controls is active
    assert node._active_plan_started_at is not None
    assert not node.cmd_pub.messages


def test_hierarchical_local_install_trusts_admitted_global_route_geometry():
    node = _install_node()
    node.use_hierarchical = True
    node._hier_phase = 'LOCAL'
    node._active_plan_request = node._capture_plan_request()
    node._fresh_request_belief = lambda _request: (
        np.zeros(3), np.eye(3), {'belief_valid': True}
    )
    node._simple_plan_safe_to_execute = lambda *_args, **_kwargs: pytest.fail(
        'hierarchical local geometry must not be vetoed after global admission'
    )

    result = node._install_control_tape(
        np.array([[.2, .1], [.2, .0]]), original_len=2,
        started_at=node.get_clock().now(),
    )

    assert result == 'installed'
    assert node._active_controls is not None


def test_a_fatal_stop_still_latches_across_a_new_request():
    node = _install_node()
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._fatal_stop_triggered = True

    node._active_plan_request = node._capture_plan_request()
    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert not _nonzero(node.cmd_pub.messages), 'fatal must remain latched'


def test_a_stop_between_acceptance_and_the_install_lock_is_still_honoured():
    """``_result_safe_to_execute`` is not the ownership barrier.

    The stop is injected after safety validation returns and before the install
    lock is taken, which is exactly the window the old code left open.
    """
    node = _install_node()
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()

    real_check = node._result_safe_to_execute

    def check_then_stop(result):
        verdict = real_check(result)
        node._publish_safe_stop_command()
        node.cmd_pub.messages.clear()
        return verdict

    node._result_safe_to_execute = check_then_stop
    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert node._active_controls is None
    assert not _nonzero(node.cmd_pub.messages)


def test_a_request_with_no_stop_still_installs_normally():
    node = _install_node()
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()

    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert node._active_controls is not None
    assert _nonzero(node.cmd_pub.messages)


def test_the_wrapper_clears_its_request_context_even_when_planning_raises():
    node = _install_node()
    node.use_hierarchical = False

    def boom():
        assert node._active_plan_request is not None
        raise RuntimeError('solver exploded')

    node._plan_once_impl = boom
    with pytest.raises(RuntimeError):
        node._plan_once()
    assert node._active_plan_request is None


# --- B. the computation-age gate --------------------------------------------

def test_a_fresh_short_tape_is_accepted_on_the_direct_path():
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()

    # Two steps of 0.25 s cover 0.5 s; the solve took 0.1 s.
    node._clock.seconds = 10.1
    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert node._active_controls is not None


def test_a_delayed_one_step_tape_is_rejected_and_fails_closed():
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()

    # One step covers 0.25 s, but the solve took 0.6 s.
    node._clock.seconds = 10.6
    node._after_plan_result(_Result([[.2, .1]]))

    assert node._active_controls is None
    assert not _nonzero(node.cmd_pub.messages)
    assert node.cmd_pub.messages, 'an over-age result must fail closed to a stop'


@pytest.mark.parametrize('elapsed_s,expect_install', [
    (.5, True),    # exactly the tape duration: the strict > keeps it
    (.5001, False),
])
def test_the_expiry_boundary_keeps_the_existing_strict_comparison(elapsed_s, expect_install):
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()

    node._clock.seconds = 10. + elapsed_s
    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert (node._active_controls is not None) is expect_install


def test_a_backward_clock_jump_during_computation_rejects_the_request():
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()

    # Time runs backwards mid-solve. No old tape exists for a timer to invalidate,
    # so the request itself must be refused.
    node._clock.seconds = 9.5
    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert node._active_controls is None


def test_a_missing_execution_context_does_not_silently_disable_validation():
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = None

    expired, why = node._plan_request_expired(None, 4)
    assert expired and why == 'missing_request_context'


def test_the_age_gate_runs_with_latency_compensation_disabled():
    """Age validation and compensation bookkeeping stay separate.

    The gate was previously reachable only through the latency-compensation
    branch, so with compensation off it never ran at all.
    """
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()
    node._clock.seconds = 11.

    node._after_plan_result(_Result([[.2, .1]]))

    assert node._active_controls is None
    # Populating age metadata must not activate the inert compensation path.
    assert node.latency_compensate_plan_handoff is False
    assert node._last_latency_skip_steps == 0


def test_delay_introduced_during_safety_validation_counts_against_the_tape():
    """The final age is read under the install lock, after validation."""
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None
    node._active_plan_request = node._capture_plan_request()

    real_check = node._result_safe_to_execute

    def slow_check(result):
        verdict = real_check(result)
        node._clock.seconds = 10.9   # validation itself took time
        return verdict

    node._result_safe_to_execute = slow_check
    node._after_plan_result(_Result([[.2, .1], [.2, .1]]))

    assert node._active_controls is None


def test_a_stale_result_cannot_issue_a_stop_that_erases_another_owners_tape():
    """Generation is checked before expiry, so cancellation wins over fail-closed."""
    node = _install_node(dt=.25)
    node.use_hierarchical = False
    node.latency_compensate_plan_handoff = False
    node._pending_plan_started_at = None

    stale_request = node._capture_plan_request()
    node._publish_safe_stop_command()

    replacement = np.array([[.1, -.2]])
    node._active_controls = replacement
    node._active_plan_started_at = _Clock(10.).now()
    node.cmd_pub.messages.clear()

    # Old, cancelled AND over-age.
    node._active_plan_request = stale_request
    node._clock.seconds = 12.
    node._after_plan_result(_Result([[.9, .9]]))

    assert node._active_controls is replacement
    assert not node.cmd_pub.messages


# --- C. the hierarchical LOCAL installer ------------------------------------
#
# The LOCAL installer assigned _pending_plan_started_at but bypassed
# _result_safe_to_execute entirely, so neither the cancellation check nor the
# age gate existed on the route the hierarchical planner actually uses. These
# drive the installer's own locked boundary directly.

def _local_install(node, controls):
    """Drive the production installation transaction the LOCAL path calls.

    Both routes now share one boundary, so these exercise real code rather than
    a copy that could drift away from it.
    """
    return node._install_control_tape(
        np.asarray(controls, dtype=float),
        original_len=int(np.asarray(controls).shape[0]),
        log_prefix='[hierarchical] local control tape expired before install',
    )


def test_local_installer_discards_work_cancelled_by_an_ordinary_stop():
    node = _install_node()
    node.use_hierarchical = True
    request = node._capture_plan_request()
    node._publish_safe_stop_command()
    node.cmd_pub.messages.clear()

    node._active_plan_request = request
    assert _local_install(node, [[.2, .1], [.2, .1]]) == 'cancelled'
    assert node._active_controls is None
    assert not _nonzero(node.cmd_pub.messages)


def test_local_installer_now_enforces_the_computation_age_gate():
    node = _install_node(dt=.25)
    node.use_hierarchical = True
    node._active_plan_request = node._capture_plan_request()
    node._clock.seconds = 10.9

    assert _local_install(node, [[.2, .1], [.2, .1]]) == 'expired'
    assert node._active_controls is None
    assert not _nonzero(node.cmd_pub.messages)


def test_the_shortened_local_safe_prefix_determines_the_tape_duration():
    """The gate must measure the prefix that installs, not the solver's output.

    A four-step solve truncated to a one-step safe prefix covers 0.25 s, not
    1.0 s, so a 0.6 s computation is over-age even though the untruncated tape
    would have been long enough to hide it.
    """
    node = _install_node(dt=.25)
    node.use_hierarchical = True
    node._active_plan_request = node._capture_plan_request()
    node._clock.seconds = 10.6

    # The full four-step tape would still be current at 0.6 s.
    expired_full, _ = node._plan_request_expired(node._active_plan_request, 4)
    assert not expired_full
    # The prefix that actually installs is not.
    assert _local_install(node, [[.2, .1]]) == 'expired'


def test_local_installer_installs_a_fresh_prefix_normally():
    node = _install_node(dt=.25)
    node.use_hierarchical = True
    node._active_plan_request = node._capture_plan_request()
    node._clock.seconds = 10.1

    assert _local_install(node, [[.2, .1], [.2, .1]]) == 'installed'
    assert node._active_controls is not None
    assert _nonzero(node.cmd_pub.messages)


def test_a_cancelled_local_result_cannot_stop_over_a_replacement():
    node = _install_node(dt=.25)
    node.use_hierarchical = True
    stale_request = node._capture_plan_request()
    node._publish_safe_stop_command()

    replacement = np.array([[.1, -.2]])
    node._active_controls = replacement
    node.cmd_pub.messages.clear()

    node._active_plan_request = stale_request
    node._clock.seconds = 12.
    assert _local_install(node, [[.9, .9]]) == 'cancelled'
    assert node._active_controls is replacement
    assert not node.cmd_pub.messages
