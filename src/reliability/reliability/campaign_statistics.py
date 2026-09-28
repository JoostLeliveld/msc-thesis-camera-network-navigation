"""Run-level statistical inference for the multi-camera fusion campaign (§16–§17).

This module is the inference backbone shared by experiments E1–E8. It is
deliberately separate from ``scripts/shared/metrics.py``: that library owns the
per-sample scoring rules (Brier, log-loss, AUROC, ECE, Spearman, coverage) and
must never be re-implemented here. This module consumes *already-computed
run-level scalars* and turns them into paired differences, clustered bootstrap
confidence intervals, Wilson intervals for binary outcomes, and the
pre-registered "beats-Toro" decision rules (§17).

Unit-of-analysis discipline (§16): the atomic observation is a run, a
route-camera-subset replay, or a fault episode — never an individual frame.
The :class:`Leaf` type therefore requires ``route``/``seed``/``episode`` keys so
that frame-level pooling is structurally discouraged and the hierarchical
bootstrap can resample by the correct clustering (route → seed → episode).

Everything here is pure and stdlib-only (deterministic given an explicit RNG
seed) so it runs in unit tests without data, ROS, or numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Mapping, Sequence


# Pre-registered primary comparisons (§16). Everything else is exploratory and
# must be labelled as such when reported.
PRIMARY_COMPARISONS = frozenset(
    {"full_vs_toro", "full_vs_gp_only", "fusion_vs_selection"}
)

_DEFAULT_Z = 1.959963984540054  # two-sided 95% normal quantile


# --------------------------------------------------------------------------- #
# Small numeric helpers (percentile / trapezoid are not scoring metrics, so
# they are allowed here; brier/logloss/auroc/ece/spearman stay in metrics.py).
# --------------------------------------------------------------------------- #
def _finite(value: object, *, field_name: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite, got {out!r}")
    return out


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile on an already-sorted sequence.

    ``q`` is a fraction in [0, 1]. Matches numpy's default 'linear' method so
    downstream figures agree with array-based reference computations.
    """

    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile over an unsorted sequence (q in [0, 1])."""

    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    return _percentile(sorted(float(v) for v in values), q)


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of an empty sequence is undefined")
    return _percentile(sorted(float(v) for v in values), 0.5)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence is undefined")
    return math.fsum(float(v) for v in values) / len(values)


# --------------------------------------------------------------------------- #
# Binary outcomes — Wilson score interval (§16 "Wilson interval").
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProportionEstimate:
    """Point estimate and Wilson score interval for a binomial proportion."""

    successes: int
    n: int
    point: float
    low: float
    high: float


def wilson_interval(successes: int, n: int, *, z: float = _DEFAULT_Z) -> ProportionEstimate:
    """Wilson score interval for a binomial success rate.

    Preferred over the normal (Wald) interval for the small run counts (§16
    notes the 20-run campaign is small) because it stays inside [0, 1] and does
    not collapse to zero width at 0/1 successes.
    """

    successes = int(successes)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = _finite(z, field_name="z")
    if z <= 0:
        raise ValueError("z must be positive")

    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return ProportionEstimate(successes=successes, n=n, point=p, low=low, high=high)


# --------------------------------------------------------------------------- #
# Paired design (§16) — identical units, Δ = proposed − baseline.
# --------------------------------------------------------------------------- #
def paired_differences(
    proposed: Mapping[str, float],
    baseline: Mapping[str, float],
) -> list[tuple[str, float]]:
    """Δ per matched unit key; requires identical key sets (paired design).

    Enforcing identical keys is the point: E1–E8 replay every method over the
    *same* recorded detections, so a key present for one method but not the
    other means a broken pairing, not a missing datum to be silently dropped.
    """

    pk = set(proposed)
    bk = set(baseline)
    if pk != bk:
        only_p = sorted(pk - bk)
        only_b = sorted(bk - pk)
        raise ValueError(
            "paired_differences requires identical unit keys; "
            f"proposed-only={only_p}, baseline-only={only_b}"
        )
    out = []
    for key in sorted(proposed):
        dp = _finite(proposed[key], field_name=f"proposed[{key}]")
        db = _finite(baseline[key], field_name=f"baseline[{key}]")
        out.append((key, dp - db))
    return out


# --------------------------------------------------------------------------- #
# Clustered / hierarchical bootstrap (§16).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Leaf:
    """One atomic paired observation for the hierarchical bootstrap.

    ``delta`` is the paired proposed−baseline difference on this leaf (or any
    per-leaf scalar the statistic consumes). ``route``/``seed``/``episode`` are
    the nesting keys; frames are never leaves.
    """

    route: str
    seed: str
    episode: str
    delta: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", str(self.route))
        object.__setattr__(self, "seed", str(self.seed))
        object.__setattr__(self, "episode", str(self.episode))
        object.__setattr__(self, "delta", _finite(self.delta, field_name="delta"))


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    low: float
    high: float
    n_boot: int
    n_units: int
    proportion_favorable: float
    lower_is_better: bool


def _favorable(delta: float, *, lower_is_better: bool) -> bool:
    # proposed − baseline. Lower-is-better metric (error): improvement is Δ<0.
    return delta < 0.0 if lower_is_better else delta > 0.0


def hierarchical_bootstrap(
    leaves: Sequence[Leaf],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    lower_is_better: bool = True,
) -> BootstrapResult:
    """Route → seed → episode nested resampling of paired deltas (§16).

    Resamples routes with replacement; within each drawn route, resamples its
    seeds with replacement; within each drawn (route, seed) run, resamples its
    episode-level leaves with replacement. This respects the experimental
    clustering instead of pretending leaves are i.i.d., which would give
    dishonestly narrow intervals.
    """

    leaves = list(leaves)
    if not leaves:
        raise ValueError("hierarchical_bootstrap needs at least one leaf")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")

    # Build the nesting: route -> seed -> [deltas].
    routes: dict[str, dict[str, list[float]]] = {}
    for leaf in leaves:
        routes.setdefault(leaf.route, {}).setdefault(leaf.seed, []).append(leaf.delta)
    route_keys = list(routes)

    rng = random.Random(seed)
    point = statistic([leaf.delta for leaf in leaves])
    replicates: list[float] = []
    for _ in range(n_boot):
        drawn: list[float] = []
        for _r in route_keys:
            route = rng.choice(route_keys)
            seed_keys = list(routes[route])
            for _s in seed_keys:
                chosen_seed = rng.choice(seed_keys)
                episodes = routes[route][chosen_seed]
                for _e in episodes:
                    drawn.append(episodes[rng.randrange(len(episodes))])
        if drawn:
            replicates.append(statistic(drawn))
    replicates.sort()
    alpha = (1.0 - ci) / 2.0
    low = _percentile(replicates, alpha)
    high = _percentile(replicates, 1.0 - alpha)
    favorable = sum(1 for leaf in leaves if _favorable(leaf.delta, lower_is_better=lower_is_better))
    return BootstrapResult(
        point=float(point),
        low=float(low),
        high=float(high),
        n_boot=len(replicates),
        n_units=len(leaves),
        proportion_favorable=favorable / len(leaves),
        lower_is_better=lower_is_better,
    )


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    lower_is_better: bool = True,
) -> BootstrapResult:
    """Flat (single-level) paired bootstrap over run-level deltas.

    Use when there is no route/seed nesting (e.g. one route, many seeds treated
    as exchangeable). For clustered data prefer :func:`hierarchical_bootstrap`.
    """

    values = [_finite(d, field_name="delta") for d in deltas]
    if not values:
        raise ValueError("paired_bootstrap needs at least one delta")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    rng = random.Random(seed)
    n = len(values)
    replicates = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        replicates.append(statistic(sample))
    replicates.sort()
    alpha = (1.0 - ci) / 2.0
    favorable = sum(1 for d in values if _favorable(d, lower_is_better=lower_is_better))
    return BootstrapResult(
        point=float(statistic(values)),
        low=float(_percentile(replicates, alpha)),
        high=float(_percentile(replicates, 1.0 - alpha)),
        n_boot=n_boot,
        n_units=n,
        proportion_favorable=favorable / n,
        lower_is_better=lower_is_better,
    )


@dataclass(frozen=True)
class PairedComparison:
    """End-to-end paired summary of proposed vs baseline over matched units."""

    n: int
    mean_delta: float
    median_delta: float
    ci_low: float
    ci_high: float
    proportion_improved: float
    lower_is_better: bool
    ci_excludes_zero: bool


def summarize_paired(
    proposed: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    lower_is_better: bool = True,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Match units, compute Δ, bootstrap the mean Δ, and summarize (§16)."""

    diffs = paired_differences(proposed, baseline)
    deltas = [d for _key, d in diffs]
    boot = paired_bootstrap(
        deltas,
        statistic=mean,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
        lower_is_better=lower_is_better,
    )
    excludes_zero = boot.low > 0.0 or boot.high < 0.0
    return PairedComparison(
        n=len(deltas),
        mean_delta=mean(deltas),
        median_delta=median(deltas),
        ci_low=boot.low,
        ci_high=boot.high,
        proportion_improved=boot.proportion_favorable,
        lower_is_better=lower_is_better,
        ci_excludes_zero=excludes_zero,
    )


# --------------------------------------------------------------------------- #
# Error-severity area under the curve (§15 E5/E6, §17 degraded rule, H3).
# --------------------------------------------------------------------------- #
def error_severity_auc(severities: Sequence[float], errors: Sequence[float]) -> float:
    """Trapezoidal area under an error-versus-failure-severity curve.

    Lower is better: a robust method's error rises more slowly with fault
    severity. Points are sorted by severity; duplicate severities are averaged
    before integration. This is not a scoring rule from metrics.py — it is a
    campaign-specific degradation summary, so it lives here.
    """

    if len(severities) != len(errors):
        raise ValueError("severities and errors must have equal length")
    if len(severities) < 2:
        raise ValueError("error_severity_auc needs at least two severity levels")
    pairs: dict[float, list[float]] = {}
    for s, e in zip(severities, errors):
        pairs.setdefault(_finite(s, field_name="severity"), []).append(
            _finite(e, field_name="error")
        )
    xs = sorted(pairs)
    if len(xs) < 2:
        raise ValueError("error_severity_auc needs at least two distinct severities")
    ys = [mean(pairs[x]) for x in xs]
    area = 0.0
    for i in range(len(xs) - 1):
        area += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return float(area)


# --------------------------------------------------------------------------- #
# Multiple-comparison discipline (§16).
# --------------------------------------------------------------------------- #
def classify_comparison(name: str) -> str:
    """'primary' for a pre-registered comparison, else 'exploratory'."""

    return "primary" if name in PRIMARY_COMPARISONS else "exploratory"


# --------------------------------------------------------------------------- #
# Pre-registered "beats-Toro" decision rules (§17). Each rule is pure logic
# over already-computed run-level aggregates and paired-CI booleans; it never
# recomputes statistics so the decision and the evidence stay decoupled and the
# rule is trivially auditable. A single secondary-metric win is NOT a claim.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NominalSummary:
    median_run_p95_ate: float
    localization_nll: float
    coverage_c95: float  # empirical χ² 95% coverage; nominal target = 0.95
    real_time_ok: bool


@dataclass(frozen=True)
class Verdict:
    passed: bool
    criteria: dict[str, bool] = field(default_factory=dict)
    notes: str = ""


def beats_toro_nominal(
    proposed: NominalSummary,
    toro: NominalSummary,
    *,
    max_error_increase_ruled_out: bool,
    nominal_coverage: float = 0.95,
) -> Verdict:
    """§17 nominal-localization rule: ALL five criteria must hold.

    ``max_error_increase_ruled_out`` is supplied by the caller from a paired
    bootstrap on the max-error metric (the CI upper bound stays below the
    pre-declared tolerance) — the rule consumes the boolean rather than
    recomputing it, keeping decision and evidence decoupled.
    """

    criteria = {
        "lower_median_p95_ate": proposed.median_run_p95_ate < toro.median_run_p95_ate,
        "lower_localization_nll": proposed.localization_nll < toro.localization_nll,
        "coverage_closer_to_nominal": (
            abs(proposed.coverage_c95 - nominal_coverage)
            < abs(toro.coverage_c95 - nominal_coverage)
        ),
        "no_meaningful_max_error_increase": bool(max_error_increase_ruled_out),
        "real_time": bool(proposed.real_time_ok),
    }
    passed = all(criteria.values())
    return Verdict(passed=passed, criteria=criteria, notes="§17 nominal localization")


def beats_toro_degraded(
    *,
    lower_error_severity_auc: bool,
    lower_max_error: bool,
    faster_fault_isolation: bool,
    lower_nav_failure_rate: bool,
) -> Verdict:
    """§17 degraded-operation rule (single-camera failure / calibration drift)."""

    criteria = {
        "lower_error_severity_auc": bool(lower_error_severity_auc),
        "lower_max_error": bool(lower_max_error),
        "faster_fault_isolation": bool(faster_fault_isolation),
        "lower_nav_failure_rate": bool(lower_nav_failure_rate),
    }
    return Verdict(
        passed=all(criteria.values()),
        criteria=criteria,
        notes="§17 degraded operation",
    )


def beats_toro_navigation(
    *,
    better_breach_free_completion: bool,
    fewer_geometry_breaches: bool,
    path_length_overhead: float,
    travel_time_overhead: float,
    physics_contacts_increase: int,
    max_path_length_overhead: float,
    max_travel_time_overhead: float,
) -> Verdict:
    """§17 navigation rule.

    Requires (better breach-free completion OR fewer geometry breaches) AND
    path-length/travel-time overheads within pre-declared tolerances AND no
    increase in physics contacts. Overheads are fractional (0.10 = +10%).
    """

    path_ok = _finite(path_length_overhead, field_name="path_length_overhead") <= _finite(
        max_path_length_overhead, field_name="max_path_length_overhead"
    )
    time_ok = _finite(travel_time_overhead, field_name="travel_time_overhead") <= _finite(
        max_travel_time_overhead, field_name="max_travel_time_overhead"
    )
    criteria = {
        "safety_improved": bool(better_breach_free_completion) or bool(fewer_geometry_breaches),
        "path_length_within_tolerance": path_ok,
        "travel_time_within_tolerance": time_ok,
        "no_physics_contact_increase": int(physics_contacts_increase) <= 0,
    }
    return Verdict(
        passed=all(criteria.values()),
        criteria=criteria,
        notes="§17 navigation",
    )
