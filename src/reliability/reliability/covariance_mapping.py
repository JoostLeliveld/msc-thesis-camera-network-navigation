"""Single reconciled trust -> update-covariance mapping (plan 07, ch.05, §10).

THIS MODULE IS THE ONE SOURCE OF TRUTH for turning a stacked trust scalar into a
filter-update covariance. It closes the known ch.05 blocker: three call sites
(planner runtime, ``single_camera_adapter``, and the offline
``geometry_visibility.trust_to_r_plan``) shared the SAME precision-blend FORMULA
but diverged only in the miss-endpoint constant (offline ``r_miss_uv=40 px`` vs
planner runtime ``r_miss_uv=120 px``). The equivalence of the formulas is proved
by a grid test (``tests/reliability/test_covariance_mapping.py``); the remaining
disagreement is purely the endpoint constant.

Two registered forms (both bounded, PSD, and monotone in trust):

* ``bounded_interpolation`` (default, plan 07 §10 preferred form)::

      R_update = R_good + (1 - tau)^gamma * (R_bad - R_good)

  ``R_good`` is the plan-06 conditional covariance (estimated, not asserted);
  ``R_bad`` is the miss-regime endpoint. Delegated to
  :func:`reliability.planning_covariance.plan_covariance`.

* ``precision_blend`` (the documented §10 ablation / ratio-inflation variant)::

      1/var = tau / r_visible^2 + (1 - tau) / r_miss^2

  Delegated to :func:`reliability.single_camera_adapter.precision_blend_covariance`
  — the exact scalar-trust blend the current planner runs.

The miss-endpoint constant is PRE-REGISTRATION-PENDING (plan 07 work item 3). No
number is blessed here: :meth:`MissEndpointPolicy.require_reconciled` raises until
the empirical decision is recorded in
``experiments/multicamera_fusion_extension/plans/DECISION_r_miss.md``. Downstream
modules (09/10) MUST take the mapping constant from here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from reliability.contracts import (
    ContractValidationError,
    UpdateCovariance,
    _finite_float,
    _finite_probability,
)
from reliability.fusion import _matrix_2x2, _validate_spd
from reliability.planning_covariance import plan_covariance
from reliability.single_camera_adapter import precision_blend_covariance

Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]

# Registered mapping forms. bounded_interpolation is the plan 07 §10 preferred
# form; precision_blend is the documented ablation.
MODE_BOUNDED = "bounded_interpolation"
MODE_PRECISION = "precision_blend"
REGISTERED_MODES = (MODE_BOUNDED, MODE_PRECISION)

# Default gate thresholds (probabilities in [0, 1]). Kept here as documented
# defaults; callers may override per study.
DEFAULT_TAU_MIN = 0.4
DEFAULT_H_MIN = 0.4

_DECISION_DOC = "experiments/multicamera_fusion_extension/plans/DECISION_r_miss.md"


@dataclass(frozen=True)
class MissEndpointPolicy:
    """The miss-regime endpoint policy — the single place the constant lives.

    ``r_visible_uv`` is the visible-regime per-axis std (px). The two candidate
    miss endpoints are carried side by side (``r_miss_uv_offline`` = 40 px from
    the geometry-visibility offline default, ``r_miss_uv_runtime`` = 120 px from
    the planner conservative default). NEITHER is blessed until the empirical
    decision (miss-regime residual tails at 4-cam operating ranges) is recorded
    and ``reconciled`` is set with ``chosen_r_miss_uv``. Until then
    :meth:`require_reconciled` raises, which FORCES the decision to be explicit
    before any R_plan number is quoted.
    """

    r_visible_uv: float = 2.5
    r_miss_uv_offline: float = 40.0
    r_miss_uv_runtime: float = 120.0
    gamma: float = 1.0
    reconciled: bool = False
    chosen_r_miss_uv: float | None = None
    provenance: str = ""

    def __post_init__(self) -> None:
        r_visible = _finite_float(self.r_visible_uv, field_name="r_visible_uv")
        if r_visible <= 0.0:
            raise ContractValidationError("r_visible_uv must be > 0")
        r_offline = _finite_float(self.r_miss_uv_offline, field_name="r_miss_uv_offline")
        r_runtime = _finite_float(self.r_miss_uv_runtime, field_name="r_miss_uv_runtime")
        if r_offline <= r_visible or r_runtime <= r_visible:
            raise ContractValidationError(
                "both miss endpoints must exceed r_visible_uv "
                "(a miss can never be more precise than a hit)"
            )
        gamma = _finite_float(self.gamma, field_name="gamma")
        if gamma <= 0.0:
            raise ContractValidationError(f"gamma must be > 0, got {gamma}")
        if self.chosen_r_miss_uv is not None:
            chosen = _finite_float(self.chosen_r_miss_uv, field_name="chosen_r_miss_uv")
            if chosen <= r_visible:
                raise ContractValidationError("chosen_r_miss_uv must exceed r_visible_uv")

    def require_reconciled(self) -> float:
        """Return the reconciled miss endpoint (px), or raise if not yet decided.

        This is the guard that keeps an ungrounded number from being quoted. It
        succeeds ONLY when ``reconciled`` is True and ``chosen_r_miss_uv`` is set
        (which happens when the pre-registration in ``DECISION_r_miss.md`` is
        filled from data).
        """

        if self.reconciled and self.chosen_r_miss_uv is not None:
            return float(self.chosen_r_miss_uv)
        raise ContractValidationError(
            "MissEndpointPolicy is not reconciled: r_miss_uv is still ambiguous "
            f"(offline={self.r_miss_uv_offline} px vs runtime={self.r_miss_uv_runtime} px). "
            "Plan 07 work item 3 forbids quoting an R_plan number until this is "
            f"decided from data and recorded in {_DECISION_DOC}. The empirical "
            "basis still owed is the miss-regime residual tails at 4-cam operating "
            "ranges (commissioning data). Set reconciled=True with chosen_r_miss_uv "
            "once that pre-registration is filled."
        )


def gate_decision(
    tau: Any,
    health: Any,
    *,
    tau_min: float = DEFAULT_TAU_MIN,
    h_min: float = DEFAULT_H_MIN,
) -> str:
    """Categorise a trust/health pair into ``accept`` | ``inflate`` | ``reject``.

    Pure function. Thresholds are probabilities in [0, 1]:

    * ``reject`` when trust or health has collapsed — ``tau < 0.5 * tau_min`` OR
      ``health < 0.5 * h_min`` (the observation is too degraded to trust at all).
    * ``accept`` when both clear their thresholds — ``tau >= tau_min`` AND
      ``health >= h_min``.
    * ``inflate`` otherwise — a usable-but-degraded regime where the update is
      taken with a larger (already trust-mapped) covariance.

    The ``reject`` band is checked first so a collapse in either signal wins.
    """

    t = _finite_probability(tau, field_name="tau")
    h = _finite_probability(health, field_name="health")
    tmin = _finite_probability(tau_min, field_name="tau_min")
    hmin = _finite_probability(h_min, field_name="h_min")
    if t < 0.5 * tmin or h < 0.5 * hmin:
        return "reject"
    if t >= tmin and h >= hmin:
        return "accept"
    return "inflate"


@dataclass(frozen=True)
class MappingDecision:
    """The logged decomposition that ``UpdateCovariance``'s closed schema cannot carry.

    ``factors`` records the per-factor contributions (spatial / confidence /
    health) that plan 07 requires logged alongside the covariance.
    """

    mode: str
    gate: str
    reason: str
    factors: dict = field(default_factory=dict)


def _as_matrix(value: Any, name: str) -> Matrix2x2:
    return _matrix_2x2(value, name)


def _trace(matrix: Matrix2x2) -> float:
    return float(matrix[0][0]) + float(matrix[1][1])


def _scale_to_miss_endpoint(cond: Matrix2x2, r_miss_uv: float) -> Matrix2x2:
    """Scale the conditional covariance up to the miss regime, preserving shape.

    A single scalar ``s = r_miss^2 / max(diag(R_cond))`` is applied so the LARGER
    per-axis std reaches ``r_miss`` exactly (both axes reach it when R_cond is
    isotropic — the common case). Scaling by a scalar keeps ``R_bad`` SPD and
    guarantees ``R_bad - R_cond`` is PSD (since ``s >= 1``), so a miss can never
    shrink covariance in any direction.
    """

    max_diag = max(float(cond[0][0]), float(cond[1][1]))
    if max_diag <= 0.0:
        raise ContractValidationError("R_cond must have positive variance on the diagonal")
    scale = (float(r_miss_uv) ** 2) / max_diag
    if scale < 1.0:
        raise ContractValidationError(
            "R_cond per-axis variance already exceeds the miss endpoint r_miss^2; "
            "cannot build a miss regime that inflates it — check units (px^2) or "
            "the reconciled r_miss_uv value"
        )
    return (
        (scale * float(cond[0][0]), scale * float(cond[0][1])),
        (scale * float(cond[1][0]), scale * float(cond[1][1])),
    )


def trust_to_update_covariance(
    tau: Any,
    health: Any,
    r_cond: Sequence[Sequence[float]],
    policy: MissEndpointPolicy,
    *,
    mode: str = MODE_BOUNDED,
    r_bad: Sequence[Sequence[float]] | None = None,
    spatial_quality: float | None = None,
    calibrated_confidence: float | None = None,
    tau_min: float = DEFAULT_TAU_MIN,
    h_min: float = DEFAULT_H_MIN,
    adapter_id: str = "",
    source: str = "covariance_mapping.v1",
    min_probability: float = 1.0e-4,
) -> tuple[UpdateCovariance, MappingDecision]:
    """Map a stacked trust scalar to an :class:`UpdateCovariance` + decision log.

    Parameters
    ----------
    tau:
        Stacked trust in [0, 1] (e.g. ``a_lcb * q_lcb * h_bar`` — already
        composed; this module does not re-compose it).
    health:
        Slow per-camera health scalar in [0, 1]; drives the gate only (kept out
        of the covariance matrix so trace stays monotone in ``tau``).
    r_cond:
        2x2 conditional covariance (px^2) = ``R_good`` (plan-06 estimate).
    policy:
        :class:`MissEndpointPolicy`; ``require_reconciled`` supplies the miss std.
    mode:
        ``bounded_interpolation`` (default, preferred) or ``precision_blend``
        (documented §10 ablation).
    r_bad:
        Optional explicit 2x2 miss-regime covariance (px^2). When omitted,
        ``R_cond`` is scaled up to the reconciled miss endpoint.
    spatial_quality, calibrated_confidence:
        Optional factor-log inputs (recorded in ``MappingDecision.factors``).

    Returns
    -------
    (UpdateCovariance, MappingDecision)
        The covariance is guaranteed SPD, exactly 2x2, finite, in px^2, and its
        trace is non-increasing in ``tau`` (lower trust never shrinks it). The
        miss endpoint is only ever read via ``policy.require_reconciled()``, so
        an unreconciled policy raises before any number is produced.
    """

    if mode not in REGISTERED_MODES:
        raise ContractValidationError(
            f"unknown mapping mode {mode!r}; registered forms are {REGISTERED_MODES}"
        )

    t = _finite_probability(tau, field_name="tau")
    r_miss_uv = policy.require_reconciled()  # forces the endpoint decision first
    gamma = float(policy.gamma)

    if mode == MODE_BOUNDED:
        cond = _as_matrix(r_cond, "r_cond")
        _validate_spd(cond, "r_cond")
        if r_bad is not None:
            bad = _as_matrix(r_bad, "r_bad")
            _validate_spd(bad, "r_bad")
        else:
            bad = _scale_to_miss_endpoint(cond, r_miss_uv)
        # plan_covariance enforces R_bad >= R_good (Loewner), SPD, gamma > 0.
        matrix = plan_covariance(t, cond, bad, gamma=gamma)
        interp_weight = (1.0 - t) ** gamma
    else:  # MODE_PRECISION
        matrix = precision_blend_covariance(
            t,
            r_visible_uv=policy.r_visible_uv,
            r_miss_uv=r_miss_uv,
            min_probability=min_probability,
        )
        interp_weight = 1.0 - t

    gate = gate_decision(t, health, tau_min=tau_min, h_min=h_min)
    available = gate != "reject"

    factors: dict = {
        "tau": t,
        "health": _finite_probability(health, field_name="health"),
        "spatial_quality": (
            None if spatial_quality is None
            else _finite_probability(spatial_quality, field_name="spatial_quality")
        ),
        "calibrated_confidence": (
            None if calibrated_confidence is None
            else _finite_probability(calibrated_confidence, field_name="calibrated_confidence")
        ),
        "gamma": gamma,
        "r_visible_uv": float(policy.r_visible_uv),
        "r_miss_uv": float(r_miss_uv),
        "interp_weight": float(interp_weight),
        "trace_px2": _trace(matrix),
    }
    reason = (
        f"{mode}: tau={t:.4f} health={factors['health']:.4f} gate={gate}; "
        f"interp between R_cond (visible) and miss endpoint std={r_miss_uv:g}px, "
        f"gamma={gamma:g}, weight={interp_weight:.4f}"
    )
    decision = MappingDecision(mode=mode, gate=gate, reason=reason, factors=factors)

    update = UpdateCovariance(
        matrix_px2=matrix,
        available=available,
        adapter_id=adapter_id,
        units="px^2",
        source=source,
        epistemic_inflation=1.0,
        staleness_inflation=1.0,
    )
    return update, decision
