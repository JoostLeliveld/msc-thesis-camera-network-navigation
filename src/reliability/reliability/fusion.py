"""Camera selection, measurement fusion and planar filter updates.

Batch fusion requires one observation per physical camera. Sequential updates
are a separate API and deliberately preserve the caller's temporal order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from statistics import median
from typing import Iterable, Mapping, Sequence

from reliability.contracts import (
    CameraObservation, CameraQuality, ContractValidationError,
    _as_pair, _as_matrix_2x2, _finite_float, _validate_spd_2x2,
)


@dataclass(frozen=True)
class MapObservation:
    camera_id: str
    timestamp_s: float
    xy_m: tuple[float, float]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    quality: CameraQuality
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ContractValidationError("camera_id must be a non-empty string")
        timestamp = _finite_float(self.timestamp_s, field_name="timestamp_s")
        if timestamp < 0.0:
            raise ContractValidationError("timestamp_s must be non-negative")
        xy = _pair(self.xy_m, "xy_m")
        cov = _matrix_2x2(self.covariance_m2, "covariance_m2")
        _validate_spd(cov, "covariance_m2")
        object.__setattr__(self, "xy_m", xy)
        object.__setattr__(self, "covariance_m2", cov)
        object.__setattr__(self, "timestamp_s", timestamp)
        if not isinstance(self.quality, CameraQuality):
            raise ContractValidationError("quality must be a CameraQuality object")
        if self.quality.camera_id and self.quality.camera_id != self.camera_id:
            raise ContractValidationError("quality.camera_id must match MapObservation.camera_id")
        if not isinstance(self.source, str):
            raise ContractValidationError("source must be a string")

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp_s": float(self.timestamp_s),
            "xy_m": [float(self.xy_m[0]), float(self.xy_m[1])],
            "covariance_m2": [
                [float(self.covariance_m2[0][0]), float(self.covariance_m2[0][1])],
                [float(self.covariance_m2[1][0]), float(self.covariance_m2[1][1])],
            ],
            "quality": self.quality.to_dict(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "MapObservation":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("MapObservation payload must be a mapping")
        missing = {"camera_id", "timestamp_s", "xy_m", "covariance_m2", "quality"} - set(payload)
        if missing:
            raise ContractValidationError(
                "MapObservation is missing fields: " + ", ".join(sorted(missing))
            )
        unknown = set(payload) - {
            "camera_id", "timestamp_s", "xy_m", "covariance_m2", "quality", "source",
        }
        if unknown:
            raise ContractValidationError(
                "MapObservation contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(
            camera_id=payload["camera_id"],
            timestamp_s=payload["timestamp_s"],
            xy_m=payload["xy_m"],
            covariance_m2=payload["covariance_m2"],
            quality=CameraQuality.from_dict(payload["quality"]),
            source=payload.get("source", ""),
        )


#: Wire format version for the per-camera map-observation batch. The planner
#: refuses a batch it does not recognise rather than silently misreading it.
MAP_OBSERVATION_BATCH_SCHEMA = "map_observation_batch/1"


def camera_measurement_batch(
    observations: Iterable[MapObservation], *, allow_empty: bool = False,
) -> tuple[MapObservation, ...]:
    """Validate camera identity before any batch selection, weighting or gate.

    Repeated cameras are an upstream event-accounting error, including identical
    duplicates. Choosing a first/last value or counting both would silently change
    the evidence. Capture-time alignment is the caller's responsibility.
    """
    items = tuple(observations)
    if not items and not allow_empty:
        raise ContractValidationError("at least one map observation is required")
    seen = set()
    for observation in items:
        if not isinstance(observation, MapObservation):
            raise ContractValidationError("batch entries must be MapObservation objects")
        if observation.camera_id in seen:
            raise ContractValidationError(f"duplicate camera in measurement batch: {observation.camera_id!r}")
        seen.add(observation.camera_id)
    return items


def _frame_id(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError("frame_id must be a non-empty string")
    return value


def map_observations_to_json(observations: Sequence[MapObservation], *, frame_id: str = "map") -> str:
    """Serialize per-camera map observations for the planner's sequential filter.

    This is the seam between projection/covariance (camera_manager's job) and
    belief updating (the planner's job). Each observation keeps its OWN
    covariance: the planner folds them in one at a time, so a per-camera
    covariance model reaches the filter intact instead of being collapsed into
    a single fused pose first.
    """
    import json

    return json.dumps(
        {
            "schema": MAP_OBSERVATION_BATCH_SCHEMA,
            "frame_id": _frame_id(frame_id),
            "observations": [obs.to_dict() for obs in camera_measurement_batch(observations, allow_empty=True)],
        },
        sort_keys=True,
        allow_nan=False,
    )


def map_observations_from_json(text: str) -> tuple[list[MapObservation], str]:
    """Inverse of :func:`map_observations_to_json`; returns (observations, frame_id)."""
    import json

    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ContractValidationError("map-observation batch must be a JSON object")
    unknown = set(payload) - {"schema", "frame_id", "observations"}
    if unknown:
        raise ContractValidationError(
            "map-observation batch contains unknown fields: " + ", ".join(sorted(unknown))
        )
    schema = str(payload.get("schema", ""))
    if schema != MAP_OBSERVATION_BATCH_SCHEMA:
        raise ContractValidationError(
            f"unsupported map-observation batch schema {schema!r}; "
            f"expected {MAP_OBSERVATION_BATCH_SCHEMA!r}"
        )
    raw = payload.get("observations")
    if not isinstance(raw, list):
        raise ContractValidationError("map-observation batch 'observations' must be a list")
    frame_id = _frame_id(payload.get("frame_id"))
    observations = camera_measurement_batch(
        (MapObservation.from_dict(item) for item in raw), allow_empty=True,
    )
    return list(observations), frame_id


@dataclass(frozen=True)
class SequentialFusionResult:
    mean_xy: tuple[float, float]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    accepted_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    nis_by_camera: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedZone:
    camera_id: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, xy: Sequence[float]) -> bool:
        x, y = _pair(xy, "xy")
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


def usable_observations(observations: Iterable[CameraObservation]) -> list[CameraObservation]:
    return [obs for obs in observations if obs.detection_valid and obs.pixel_uv is not None]


def select_primary_camera(
    observations: Iterable[CameraObservation],
    *,
    primary_camera_id: str = "camera_A",
) -> CameraObservation | None:
    valid = usable_observations(observations)
    for obs in valid:
        if obs.camera_id == primary_camera_id:
            return obs
    return min(valid, key=lambda obs: obs.camera_id) if valid else None


def select_fixed_zone(
    observations: Iterable[CameraObservation],
    *,
    state_xy: Sequence[float],
    zones: Sequence[FixedZone],
) -> CameraObservation | None:
    valid = usable_observations(observations)
    for zone in zones:
        if not zone.contains(state_xy):
            continue
        for obs in valid:
            if obs.camera_id == zone.camera_id:
                return obs
    return min(valid, key=lambda obs: obs.camera_id) if valid else None


def select_highest_detector_score(observations: Iterable[CameraObservation]) -> CameraObservation | None:
    valid = usable_observations(observations)
    if not valid:
        return None
    return min(
        valid,
        key=lambda obs: (-obs.detector_score, obs.measurement_age_s, obs.camera_id),
    )


def select_freshest_valid(observations: Iterable[CameraObservation]) -> CameraObservation | None:
    valid = usable_observations(observations)
    if not valid:
        return None
    return min(
        valid,
        key=lambda obs: (obs.measurement_age_s, -obs.detector_score, obs.camera_id),
    )


def select_best_static_reliability(
    observations: Iterable[CameraObservation],
    qualities: Mapping[str, CameraQuality],
) -> CameraObservation | None:
    valid = usable_observations(observations)
    if not valid:
        return None
    return min(
        valid,
        key=lambda obs: (
            -qualities.get(obs.camera_id, obs.quality()).p_available,
            -obs.detector_score,
            obs.camera_id,
        ),
    )


def conservative_camera_score(
    observation: CameraObservation,
    quality: CameraQuality | None = None,
    *,
    lambda_age: float = 1.0,
    gamma_epistemic: float = 1.0,
) -> float:
    quality = quality or observation.quality()
    age = max(float(observation.measurement_age_s), 0.0)
    return float(
        quality.p_available
        * quality.association_confidence
        * math.exp(-float(lambda_age) * age)
        * math.exp(-float(gamma_epistemic) * float(quality.epistemic_score))
    )


def select_conservative_best_camera(
    observations: Iterable[CameraObservation],
    qualities: Mapping[str, CameraQuality] | None = None,
    *,
    lambda_age: float = 1.0,
    gamma_epistemic: float = 1.0,
) -> CameraObservation | None:
    qualities = qualities or {}
    valid = usable_observations(observations)
    if not valid:
        return None
    return min(
        valid,
        key=lambda obs: (
            -conservative_camera_score(
                obs,
                qualities.get(obs.camera_id),
                lambda_age=lambda_age,
                gamma_epistemic=gamma_epistemic,
            ),
            obs.camera_id,
        ),
    )


def sequential_kalman_update_2d(
    prior_mean_xy: Sequence[float],
    prior_cov_m2: Sequence[Sequence[float]],
    observations: Iterable[MapObservation],
    *,
    nis_gate: float | None = 9.21,
    disagreement_gate_m: float | None = None,
    belief_floors: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> SequentialFusionResult:
    """Apply valid camera map observations sequentially with per-camera gates.

    `belief_floors` maps camera id to that sighting's bias floor
    (`reliability.bias_floor.bias_floor_matrix`, D5 layer 3). Floors of the cameras that
    were ACCEPTED are combined by `combine_floors` and applied once to the final
    posterior, which keeps the result independent of the order observations arrive in --
    flooring after each fold would leave the belief at the mercy of whichever camera came
    last. A camera the gates rejected contributes no floor, because it contributed no
    information to bound.

    Default `None`, so behaviour is unchanged unless a caller asks: the floor changes what
    the filter is allowed to believe, which is a method decision rather than a default.
    """

    mean = _pair(prior_mean_xy, "prior_mean_xy")
    cov = _matrix_2x2(prior_cov_m2, "prior_cov_m2")
    _validate_spd(cov, "prior_cov_m2")
    accepted: list[str] = []
    rejected: list[str] = []
    nis_by_camera: dict[str, float] = {}

    for obs in observations:
        z = obs.xy_m
        if disagreement_gate_m is not None:
            residual_norm = math.hypot(z[0] - mean[0], z[1] - mean[1])
            if residual_norm > float(disagreement_gate_m):
                rejected.append(obs.camera_id)
                nis_by_camera[obs.camera_id] = math.nan
                continue
        innovation = (z[0] - mean[0], z[1] - mean[1])
        s_mat = _mat_add(cov, obs.covariance_m2)
        nis = _quad_form_inverse_2x2(innovation, s_mat)
        nis_by_camera[obs.camera_id] = nis
        if nis_gate is not None and math.isfinite(nis) and nis > float(nis_gate):
            rejected.append(obs.camera_id)
            continue
        k = _mat_mul(cov, _mat_inv_2x2(s_mat))
        delta = _mat_vec(k, innovation)
        mean = (mean[0] + delta[0], mean[1] + delta[1])
        cov = _joseph_covariance(cov, obs.covariance_m2, k)
        accepted.append(obs.camera_id)

    if belief_floors and accepted:
        from reliability.bias_floor import apply_belief_floor, combine_camera_floors

        floor = combine_camera_floors(belief_floors, accepted)
        if floor is not None:
            cov = _symmetrize(apply_belief_floor(cov, floor))

    return SequentialFusionResult(
        mean_xy=mean,
        covariance_m2=cov,
        accepted_camera_ids=tuple(accepted),
        rejected_camera_ids=tuple(rejected),
        nis_by_camera=nis_by_camera,
    )


def independent_measurement_fusion_2d(
    observations: Iterable[MapObservation],
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    """Fuse independent 2-D measurements without inventing a state prior.

    ``R_fused = (sum R_i^-1)^-1`` and
    ``z_fused = R_fused sum R_i^-1 z_i``.  This produces a measurement for the
    planner's one real belief filter.  It must not be replaced by a Kalman
    update against an arbitrary identity prior, which would filter the cameras
    once here and then filter the resulting pseudo-measurement a second time.
    Cross-camera correlation is deliberately outside this historical baseline.
    """

    items = camera_measurement_batch(observations)
    information = ((0.0, 0.0), (0.0, 0.0))
    information_vector = (0.0, 0.0)
    for observation in items:
        precision = _mat_inv_2x2(observation.covariance_m2)
        information = _mat_add(information, precision)
        weighted = _mat_vec(precision, observation.xy_m)
        information_vector = (
            information_vector[0] + weighted[0],
            information_vector[1] + weighted[1],
        )
    covariance = _symmetrize(_mat_inv_2x2(_symmetrize(information)))
    mean = _mat_vec(covariance, information_vector)
    _validate_spd(covariance, "fused_measurement_covariance_m2")
    return mean, covariance


def select_smallest_covariance(
    observations: Iterable[MapObservation],
) -> MapObservation | None:
    """F1: the single most precise camera at this instant, by trace of its covariance.

    The honest simple baseline for the fusion comparison: if one good camera is as good as
    the network, the network claim is dead and the paper should say so.

    Distinct from :func:`select_information_best`, which scores an observation against the
    filter's current prior and therefore answers "which helps most now?" rather than "which
    is most precise?". This one needs no prior, so it can be stated as a rule a reader can
    reimplement. Equal traces are resolved by camera ID, independently of callback order.
    """

    return min(
        camera_measurement_batch(observations, allow_empty=True),
        key=lambda obs: (obs.covariance_m2[0][0] + obs.covariance_m2[1][1], obs.camera_id),
        default=None,
    )


def distance_angle_weights(
    observations: Sequence[MapObservation],
    camera_positions_m: Mapping[str, Sequence[float]],
    *,
    epsilon: float = 1e-6,
) -> dict:
    """F2's weights: nearer cameras with a steeper look count more.

    ``q_c = max(cos alpha_c, 0) / (d_c^2 + eps)``, normalised. ``d_c`` is the horizontal
    range from the camera to the reported position and ``alpha_c`` is the angle between the
    camera's line of sight and the floor's normal, so a grazing view scores near zero and a
    steep one near one.

    This is the intuitive engineering answer, and it is deliberately FROZEN as written: its
    job in the comparison is to be the honest simple baseline, not a tuned competitor. It
    ignores the covariance entirely, which is the point -- knowing where the cameras are is
    not the same as knowing how good their readings are.
    """

    items = camera_measurement_batch(observations)
    epsilon = _finite_float(epsilon, field_name="epsilon")
    if epsilon <= 0.0:
        raise ContractValidationError("epsilon must be positive")
    scores = {}
    for obs in items:
        position = camera_positions_m.get(obs.camera_id)
        if position is None:
            raise ContractValidationError(
                f"no camera position for {obs.camera_id!r}; refusing to guess one"
            )
        if len(position) != 3:
            raise ContractValidationError(f"camera position for {obs.camera_id!r} must have three coordinates")
        cx, cy, cz = (_finite_float(value, field_name=f"camera_positions_m[{obs.camera_id!r}]")
                      for value in position)
        dx, dy = obs.xy_m[0] - cx, obs.xy_m[1] - cy
        horizontal_sq = dx * dx + dy * dy
        slant = math.sqrt(horizontal_sq + cz * cz)
        cos_alpha = abs(cz) / slant if slant > 0.0 else 0.0
        scores[obs.camera_id] = max(cos_alpha, 0.0) / (horizontal_sq + epsilon)
    total = sum(scores.values())
    if total <= 0.0:
        # every camera grazing: fall back to equal weights rather than dividing by zero
        return {obs.camera_id: 1.0 / len(items) for obs in items}
    return {camera_id: score / total for camera_id, score in scores.items()}


def distance_angle_weighted_fusion_2d(
    observations: Sequence[MapObservation],
    camera_positions_m: Mapping[str, Sequence[float]],
    *,
    epsilon: float = 1e-6,
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    """F2: combine the positions with :func:`distance_angle_weights`.

    The covariance reported is that of the weighted mean under independence,
    ``sum w_c^2 Sigma_c``, so this arm can be scored for honesty exactly like the others
    rather than being excused from the test.
    """

    items = tuple(observations)
    weights = distance_angle_weights(items, camera_positions_m, epsilon=epsilon)
    mean = (0.0, 0.0)
    covariance = ((0.0, 0.0), (0.0, 0.0))
    for obs in items:
        w = weights[obs.camera_id]
        mean = (mean[0] + w * obs.xy_m[0], mean[1] + w * obs.xy_m[1])
        covariance = _mat_add(covariance, _mat_scale(obs.covariance_m2, w * w))
    covariance = _symmetrize(covariance)
    _validate_spd(covariance, "distance_angle_fused_covariance_m2")
    return mean, covariance


def joint_network_estimate_2d(
    observations: Sequence[MapObservation],
    *,
    huber_delta: float = 2.5,
    max_iterations: int = 25,
    convergence_m: float = 1.0e-9,
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
    """F4: estimate once from the simultaneous camera batch, then Gaussianise.

    The old F4 multiplied per-camera Gaussian densities with exponents ``1/N``. That made
    its point estimate identical to independent fusion and changed only the covariance by
    a fixed factor. This estimator instead treats the admitted camera batch as one
    distributed sensor:

    1. solve one robust generalized least-squares problem over all equivalent position
       residuals using Huber IRLS;
    2. *after* that point has been found, form one network-level Gaussian approximation
       with a finite-sample sandwich covariance.

    The sandwich contains both each view's propagated detector uncertainty and the
    between-view residual outer products. Consequently identical equal-quality readings
    retain one camera's covariance rather than shrinking by ``1/N``, while disagreement
    widens the network ellipse in the direction in which the cameras disagree. This is a
    runtime covariance model, not a claim of calibration: its scale still has to be frozen
    on commissioning data and tested on held-out routes.
    """

    items = camera_measurement_batch(observations)
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ContractValidationError("huber_delta must be finite and positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations <= 0:
        raise ContractValidationError("max_iterations must be a positive integer")
    iterations = int(max_iterations)
    tolerance = float(convergence_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ContractValidationError("convergence_m must be finite and non-negative")
    if len(items) == 1:
        return items[0].xy_m, items[0].covariance_m2

    # A component-wise median gives the robust solve a start that a single overconfident
    # camera cannot choose. Sorting also makes the result independent of message order.
    xs = sorted(obs.xy_m[0] for obs in items)
    ys = sorted(obs.xy_m[1] for obs in items)

    mean = (median(xs), median(ys))
    precisions = [_mat_inv_2x2(obs.covariance_m2) for obs in items]
    for _ in range(iterations):
        information = ((0.0, 0.0), (0.0, 0.0))
        information_vector = (0.0, 0.0)
        for obs, precision in zip(items, precisions):
            residual = (obs.xy_m[0] - mean[0], obs.xy_m[1] - mean[1])
            distance = math.sqrt(max(_quad_form(residual, precision), 0.0))
            weight = 1.0 if distance <= delta else delta / distance
            weighted_precision = _mat_scale(precision, weight)
            information = _mat_add(information, weighted_precision)
            contribution = _mat_vec(weighted_precision, obs.xy_m)
            information_vector = (
                information_vector[0] + contribution[0],
                information_vector[1] + contribution[1],
            )
        information = _symmetrize(information)
        updated = _mat_vec(_mat_inv_2x2(information), information_vector)
        if math.hypot(updated[0] - mean[0], updated[1] - mean[1]) <= tolerance:
            mean = updated
            break
        mean = updated

    # Re-evaluate the weights at the converged point. B is the robust score's second
    # moment. Multiplying the sandwich by the effective batch size removes the usual
    # independent-sample contraction: this batch is one sensor report, not N reports.
    information = ((0.0, 0.0), (0.0, 0.0))
    score_second_moment = ((0.0, 0.0), (0.0, 0.0))
    weights = []
    for obs, precision in zip(items, precisions):
        residual = (obs.xy_m[0] - mean[0], obs.xy_m[1] - mean[1])
        distance = math.sqrt(max(_quad_form(residual, precision), 0.0))
        weight = 1.0 if distance <= delta else delta / distance
        weights.append(weight)
        information = _mat_add(information, _mat_scale(precision, weight))
        local_second_moment = _mat_add(obs.covariance_m2, _outer(residual))
        contribution = _mat_mul(_mat_mul(precision, local_second_moment), precision)
        score_second_moment = _mat_add(
            score_second_moment, _mat_scale(contribution, weight * weight)
        )
    information_inverse = _mat_inv_2x2(_symmetrize(information))
    effective_batch_size = sum(weights) ** 2 / sum(weight * weight for weight in weights)
    covariance = _mat_scale(
        _mat_mul(_mat_mul(information_inverse, score_second_moment), information_inverse),
        effective_batch_size,
    )
    covariance = _symmetrize(covariance)
    _validate_spd(covariance, "joint_network_covariance_m2")
    return mean, covariance


def camera_disagreement_m(observations: Sequence[MapObservation]) -> float:
    if len(observations) < 2:
        return 0.0
    max_dist = 0.0
    for i, obs_a in enumerate(observations):
        for obs_b in observations[i + 1 :]:
            max_dist = max(
                max_dist,
                math.hypot(obs_a.xy_m[0] - obs_b.xy_m[0], obs_a.xy_m[1] - obs_b.xy_m[1]),
            )
    return float(max_dist)


@dataclass(frozen=True)
class FuseOrSelectDecision:
    """Outcome of the fuse-vs-select consistency logic (plan 09, §12.5)."""

    mode: str
    selected: tuple[str, ...]
    excluded: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.mode not in ("fuse", "select", "none"):
            raise ContractValidationError("mode must be one of 'fuse', 'select', 'none'")
        object.__setattr__(self, "selected", tuple(str(c) for c in self.selected))
        object.__setattr__(self, "excluded", tuple(str(c) for c in self.excluded))


def joseph_update_2d(
    state_xy: Sequence[float],
    cov_xy: Sequence[Sequence[float]],
    observation: MapObservation,
    *,
    belief_floor: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]], float]:
    """Single position update in Joseph form: (I−KH)P(I−KH)ᵀ + KRKᵀ, H = I.

    Numerically stable at small R. Returns (state, covariance, nis) where
    nis is the innovation Mahalanobis distance d² = νᵀ S⁻¹ ν, S = P + R.

    `belief_floor` is the optional bias floor of `DESIGN_LOCK.md` D5 layer 3, built by
    `reliability.bias_floor.bias_floor_matrix` from the sighting's range and bearing. It
    is applied to the POSTERIOR, which is the only place it does what it is for: added to
    `R` instead, a persistent error would be treated as fresh noise on every frame and the
    filter would still shrink it like 1/N, just from a larger start.

    Default `None`, so behaviour is unchanged unless a caller asks for it -- the floor
    changes what the filter is allowed to believe and that is a method decision, not a
    default that should arrive silently with a library upgrade.
    """

    mean = _pair(state_xy, "state_xy")
    cov = _matrix_2x2(cov_xy, "cov_xy")
    _validate_spd(cov, "cov_xy")
    r_mat = observation.covariance_m2

    z = observation.xy_m
    innovation = (z[0] - mean[0], z[1] - mean[1])
    s_mat = _mat_add(cov, r_mat)
    nis = _quad_form_inverse_2x2(innovation, s_mat)
    k = _mat_mul(cov, _mat_inv_2x2(s_mat))
    delta = _mat_vec(k, innovation)
    mean_post = (mean[0] + delta[0], mean[1] + delta[1])
    cov_post = _joseph_covariance(cov, r_mat, k)
    if belief_floor is not None:
        from reliability.bias_floor import apply_belief_floor

        cov_post = _symmetrize(apply_belief_floor(cov_post, _matrix_2x2(
            belief_floor, "belief_floor")))
    return mean_post, cov_post, float(nis)


def robust_reweight_covariance(
    covariance: Sequence[Sequence[float]],
    nis: float,
    *,
    dof: float = 4.0,
    w_min: float = 0.1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Student-t robust reweighting of an observation covariance.

    w = (dof + 2) / (dof + nis); returns R / clamp(w, w_min, 1.0).

    Deliberate deviation from plan 09's literal ``R / max(w, w_min)``: the
    weight is additionally capped at 1.0 so the mapping is monotone
    *inflating* only. Uncapped, nis < 2 would give w > 1 and shrink R
    ("sharpening"), violating plan 07's property that lower trust never
    reduces covariance — and a suspiciously-perfect observation should not
    be trusted beyond its nominal R.  Limits: nis → 0 ⇒ R unchanged;
    nis → ∞ ⇒ R / w_min floor. Ablatable: skipping this call = plain update.
    """

    r_mat = _matrix_2x2(covariance, "covariance")
    _validate_spd(r_mat, "covariance")
    nis_value = float(nis)
    if not math.isfinite(nis_value) or nis_value < 0.0:
        raise ContractValidationError("nis must be finite and non-negative")
    dof_value = float(dof)
    if not math.isfinite(dof_value) or dof_value <= 0.0:
        raise ContractValidationError("dof must be finite and positive")
    w_min_value = float(w_min)
    if not math.isfinite(w_min_value) or not 0.0 < w_min_value <= 1.0:
        raise ContractValidationError("w_min must be in (0, 1]")

    weight = (dof_value + 2.0) / (dof_value + nis_value)
    weight = min(max(weight, w_min_value), 1.0)
    scale = 1.0 / weight
    return (
        (r_mat[0][0] * scale, r_mat[0][1] * scale),
        (r_mat[1][0] * scale, r_mat[1][1] * scale),
    )


def expected_information_gain(
    cov_prior: Sequence[Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
) -> float:
    """Expected information gain Δ = log|P⁻| − log|P⁺| of a position update.

    P⁺ is the standard posterior for H = I: P⁺ = (I − K)P⁻ with
    K = P⁻ (P⁻ + R)⁻¹ (equivalently (P⁻¹ + R⁻¹)⁻¹). Always > 0 for SPD inputs.
    """

    p_mat = _matrix_2x2(cov_prior, "cov_prior")
    _validate_spd(p_mat, "cov_prior")
    r_mat = _matrix_2x2(observation_covariance, "observation_covariance")
    _validate_spd(r_mat, "observation_covariance")

    # det(P)/det(P+) = det(P+R)/det(R). Avoid subtracting two almost
    # equal matrices to obtain a tiny posterior, and avoid determinant underflow.
    return _logdet_spd_2x2(_mat_add(p_mat, r_mat)) - _logdet_spd_2x2(r_mat)


def select_information_best(
    observations: Sequence[MapObservation],
    cov_prior: Sequence[Sequence[float]],
) -> MapObservation | None:
    """B8: pick the observation with the largest expected information gain.

    Geometry-aware: a high-reliability observation with poor geometry
    (large variance along the prior's uncertain axis) can lose to a less
    reliable one that constrains the right direction.
    """

    p_mat = _matrix_2x2(cov_prior, "cov_prior")
    _validate_spd(p_mat, "cov_prior")
    return min(
        camera_measurement_batch(observations, allow_empty=True),
        key=lambda obs: (-expected_information_gain(p_mat, obs.covariance_m2), obs.camera_id),
        default=None,
    )


def fuse_or_select(
    observations: Sequence[MapObservation],
    taus: Mapping[str, float],
    healths: Mapping[str, float],
    *,
    tau_min: float,
    h_min: float,
    disagreement_gate_m: float,
) -> FuseOrSelectDecision:
    """Fuse-vs-select consistency logic (plan 09, §12.5).

    Candidate set C = {i : τ_i ≥ tau_min and h_i ≥ h_min} (a camera missing
    from ``taus``/``healths`` is treated as 0.0, i.e. excluded).

    - |C| = 0 → mode 'none'; |C| = 1 → mode 'select'.
    - If all pairwise position disagreements within C are ≤
      ``disagreement_gate_m`` → mode 'fuse' with all of C.
    - Otherwise iteratively drop the camera involved in the most
      above-gate pairs (ties broken by lowest τ, then camera_id for
      determinism) and re-check, until the remainder is mutually
      consistent (fuse) or a single camera remains (select).
    """

    tau_min_value = _require_probability(tau_min, "tau_min")
    h_min_value = _require_probability(h_min, "h_min")
    gate = float(disagreement_gate_m)
    if not math.isfinite(gate) or gate <= 0.0:
        raise ContractValidationError("disagreement_gate_m must be finite and positive")

    excluded: list[str] = []
    candidates: list[MapObservation] = []
    validated_taus: dict[str, float] = {}
    for obs in observations:
        tau = _require_probability(
            taus.get(obs.camera_id, 0.0),
            f"taus[{obs.camera_id!r}]",
        )
        health = _require_probability(
            healths.get(obs.camera_id, 0.0),
            f"healths[{obs.camera_id!r}]",
        )
        validated_taus[obs.camera_id] = tau
        if tau >= tau_min_value and health >= h_min_value:
            candidates.append(obs)
        else:
            excluded.append(obs.camera_id)

    if not candidates:
        return FuseOrSelectDecision(
            mode="none",
            selected=(),
            excluded=tuple(excluded),
            reason="no candidate passed tau_min/h_min thresholds",
        )
    if len(candidates) == 1:
        return FuseOrSelectDecision(
            mode="select",
            selected=(candidates[0].camera_id,),
            excluded=tuple(excluded),
            reason="single candidate above thresholds",
        )

    dropped_for_disagreement: list[str] = []
    while len(candidates) > 1:
        violations: dict[str, int] = {obs.camera_id: 0 for obs in candidates}
        any_violation = False
        for i, obs_a in enumerate(candidates):
            for obs_b in candidates[i + 1 :]:
                dist = math.hypot(
                    obs_a.xy_m[0] - obs_b.xy_m[0],
                    obs_a.xy_m[1] - obs_b.xy_m[1],
                )
                if dist > gate:
                    any_violation = True
                    violations[obs_a.camera_id] += 1
                    violations[obs_b.camera_id] += 1
        if not any_violation:
            break
        worst_id = min(
            violations,
            key=lambda cid: (-violations[cid], validated_taus[cid], cid),
        )
        dropped_for_disagreement.append(worst_id)
        excluded.append(worst_id)
        candidates = [obs for obs in candidates if obs.camera_id != worst_id]

    if len(candidates) == 1:
        return FuseOrSelectDecision(
            mode="select",
            selected=(candidates[0].camera_id,),
            excluded=tuple(excluded),
            reason=(
                "pairwise disagreement above gate; dropped "
                + ", ".join(dropped_for_disagreement)
            ),
        )
    reason = "all candidates mutually consistent"
    if dropped_for_disagreement:
        reason = (
            "consistent after dropping disagreeing camera(s): "
            + ", ".join(dropped_for_disagreement)
        )
    return FuseOrSelectDecision(
        mode="fuse",
        selected=tuple(obs.camera_id for obs in candidates),
        excluded=tuple(excluded),
        reason=reason,
    )


def _require_probability(value: object, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"{field_name} must be a finite probability in [0, 1]"
        ) from exc
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ContractValidationError(
            f"{field_name} must be a finite probability in [0, 1]"
        )
    return out


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    return _as_pair(value, field_name=field_name)


def _matrix_2x2(value: Sequence[Sequence[float]], field_name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    return _as_matrix_2x2(value, field_name=field_name)


def _validate_spd(matrix: tuple[tuple[float, float], tuple[float, float]], field_name: str) -> None:
    _validate_spd_2x2(matrix, field_name=field_name)


def _mat_scale(a, k):
    return ((a[0][0] * k, a[0][1] * k), (a[1][0] * k, a[1][1] * k))


def _mat_add(a, b):
    return ((a[0][0] + b[0][0], a[0][1] + b[0][1]), (a[1][0] + b[1][0], a[1][1] + b[1][1]))


def _mat_sub(a, b):
    return ((a[0][0] - b[0][0], a[0][1] - b[0][1]), (a[1][0] - b[1][0], a[1][1] - b[1][1]))


def _mat_mul(a, b):
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def _mat_vec(a, v):
    return (a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1])


def _mat_inv_2x2(a):
    scale = max(abs(value) for row in a for value in row)
    if not math.isfinite(scale) or scale == 0.0:
        raise ContractValidationError("matrix must be finite and invertible")
    aa, ab = a[0][0] / scale, a[0][1] / scale
    ac, ad = a[1][0] / scale, a[1][1] / scale
    det = aa * ad - ab * ac
    if not math.isfinite(det) or det == 0.0:
        raise ContractValidationError("matrix is singular")
    inverse = ((ad / det / scale, -ab / det / scale),
               (-ac / det / scale, aa / det / scale))
    if not all(math.isfinite(value) for row in inverse for value in row):
        raise ContractValidationError("matrix inverse is not finite")
    return inverse


def _logdet_spd_2x2(matrix):
    scale = max(abs(value) for row in matrix for value in row)
    determinant = ((matrix[0][0] / scale) * (matrix[1][1] / scale)
                   - (matrix[0][1] / scale) * (matrix[1][0] / scale))
    return 2.0 * math.log(scale) + math.log(determinant)


def _joseph_covariance(prior, measurement, gain):
    """One stable posterior formula for both planar update APIs."""
    i_minus_k = _mat_sub(((1.0, 0.0), (0.0, 1.0)), gain)
    return _symmetrize(_mat_add(
        _mat_mul(_mat_mul(i_minus_k, prior), _transpose(i_minus_k)),
        _mat_mul(_mat_mul(gain, measurement), _transpose(gain)),
    ))


def _quad_form_inverse_2x2(v, a):
    inv = _mat_inv_2x2(a)
    iv = _mat_vec(inv, v)
    return float(v[0] * iv[0] + v[1] * iv[1])


def _quad_form(v, a):
    av = _mat_vec(a, v)
    return float(v[0] * av[0] + v[1] * av[1])


def _outer(v):
    return ((v[0] * v[0], v[0] * v[1]), (v[1] * v[0], v[1] * v[1]))


def _transpose(a):
    return ((a[0][0], a[1][0]), (a[0][1], a[1][1]))


def _det_2x2(a):
    return float(a[0][0] * a[1][1] - a[0][1] * a[1][0])


def _symmetrize(a):
    off = 0.5 * (a[0][1] + a[1][0])
    return ((float(a[0][0]), float(off)), (float(off), float(a[1][1])))
