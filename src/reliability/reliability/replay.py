"""Offline replay core for camera-observation estimator comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence

from reliability.camera_manager import CameraManager, CameraManagerConfig
from reliability.contracts import CameraQuality, ContractValidationError
from reliability.dynamic_occlusion import (
    DynamicActorState,
    DynamicOcclusionConfig,
    dynamic_occlusion_adjusted_observation,
)
from reliability.fusion import MapObservation, SequentialFusionResult, sequential_kalman_update_2d
from reliability.handover import HandoverUncertaintyConfig, HandoverUncertaintyDiagnostic, handover_adjusted_observation
from reliability.health_ewma import (
    HealthDebouncer,
    HealthDebouncerConfig,
    InnovationHealthConfig,
    InnovationHealthMonitor,
)
from reliability.providers import CameraReliabilityProvider


class ReplayMode(str, Enum):
    ODOM_ONLY = "R0_odom_only"
    SINGLE_FIXED_R = "R1_single_fixed_R"
    SINGLE_FIXED_R_NIS = "R2_single_fixed_R_NIS"
    DETECTOR_SCORE_R = "R3_detector_score_R"
    CURRENT_GP_R = "R4_current_GP_R"
    SEQUENTIAL_FUSION = "M5_sequential_fusion"
    CONSERVATIVE_SELECTION = "M6_conservative_selection"
    HANDOVER_AWARE_SELECTION = "M7_handover_aware_selection"
    HYSTERETIC_HANDOVER_SELECTION = "M8_hysteretic_handover_selection"
    HEALTH_AWARE_FUSION = "B6_health_aware_fusion"
    DYNAMIC_OCCLUSION_AWARE_FUSION = "D1_dynamic_occlusion_aware_fusion"


@dataclass(frozen=True)
class ReplayFrame:
    timestamp_s: float
    odometry_xy_m: tuple[float, float]
    observations: tuple[MapObservation, ...] = field(default_factory=tuple)
    dynamic_actors: tuple[DynamicActorState, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "odometry_xy_m", _pair(self.odometry_xy_m, "odometry_xy_m"))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "dynamic_actors", tuple(self.dynamic_actors))


@dataclass(frozen=True)
class EvaluationFrame:
    timestamp_s: float
    truth_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "truth_xy_m", _pair(self.truth_xy_m, "truth_xy_m"))


@dataclass(frozen=True)
class ReplayConfig:
    mode: ReplayMode
    initial_cov_m2: tuple[tuple[float, float], tuple[float, float]] = ((0.05**2, 0.0), (0.0, 0.05**2))
    process_noise_m2: tuple[tuple[float, float], tuple[float, float]] = ((0.012**2, 0.0), (0.0, 0.012**2))
    fixed_measurement_cov_m2: tuple[tuple[float, float], tuple[float, float]] = ((0.08**2, 0.0), (0.0, 0.08**2))
    nis_gate: float | None = None
    detector_score_min_cov_m2: float = 0.04**2
    detector_score_max_cov_m2: float = 0.40**2
    max_error_divergence_m: float = 1.0
    quality_providers: Mapping[str, CameraReliabilityProvider] = field(default_factory=dict)
    handover_config: HandoverUncertaintyConfig = field(default_factory=HandoverUncertaintyConfig)
    camera_manager_config: CameraManagerConfig = field(default_factory=CameraManagerConfig)
    # B6 health-aware fusion (HEALTH_AWARE_FUSION mode) — see _health_aware_observations.
    health_config: InnovationHealthConfig | None = None
    health_debouncer_config: HealthDebouncerConfig | None = None
    # health below this = "inconsistent" for the debouncer. Nominal health is
    # ~sigmoid(eta0)=0.95, so 0.7 flags a meaningful drop while sparing healthy
    # cameras. PROVISIONAL default — the fault-containment pre-registration
    # finalizes it (and eta/rho) on the pilot before the confirmatory campaign.
    health_inflate_threshold: float = 0.7
    health_suspect_inflation: float = 9.0
    # D1 dynamic-occlusion-aware fusion. Actor states must come from an
    # operational tracker at the same frame timestamp, never simulation truth.
    dynamic_occlusion_config: DynamicOcclusionConfig | None = None


@dataclass(frozen=True)
class ReplayStep:
    timestamp_s: float
    mean_xy_m: tuple[float, float]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    accepted_camera_ids: tuple[str, ...]
    rejected_camera_ids: tuple[str, ...]
    nis_by_camera: dict[str, float]
    handover_diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    camera_manager_decision: dict[str, Any] = field(default_factory=dict)
    health_by_camera: dict[str, float] = field(default_factory=dict)
    health_state_by_camera: dict[str, str] = field(default_factory=dict)
    dynamic_occlusion_diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReplayMetrics:
    rmse_m: float
    p95_error_m: float
    max_error_m: float
    final_error_m: float
    mean_nis: float
    mean_nees: float
    covariance_1sigma_coverage: float
    covariance_2sigma_coverage: float
    update_acceptance_rate: float
    divergence_count: int
    handover_count: int
    unqualified_handover_count: int


@dataclass(frozen=True)
class ReplayResult:
    config: ReplayConfig
    steps: tuple[ReplayStep, ...]
    metrics: ReplayMetrics | None = None


def required_replay_configs(
    *,
    quality_providers: Mapping[str, CameraReliabilityProvider] | None = None,
) -> tuple[ReplayConfig, ...]:
    providers = dict(quality_providers or {})
    return (
        ReplayConfig(mode=ReplayMode.ODOM_ONLY),
        ReplayConfig(mode=ReplayMode.SINGLE_FIXED_R),
        ReplayConfig(mode=ReplayMode.SINGLE_FIXED_R_NIS, nis_gate=9.21),
        ReplayConfig(mode=ReplayMode.DETECTOR_SCORE_R, nis_gate=9.21),
        ReplayConfig(mode=ReplayMode.CURRENT_GP_R, nis_gate=9.21, quality_providers=providers),
    )


def run_replay(
    frames: Sequence[ReplayFrame],
    config: ReplayConfig,
    *,
    evaluation_frames: Sequence[EvaluationFrame] | None = None,
) -> ReplayResult:
    if not frames:
        return ReplayResult(config=config, steps=tuple(), metrics=None)

    ordered = sorted(frames, key=lambda frame: frame.timestamp_s)
    mean = ordered[0].odometry_xy_m
    cov = _matrix(config.initial_cov_m2, "initial_cov_m2")
    previous_odom = ordered[0].odometry_xy_m
    steps: list[ReplayStep] = []
    accepted_total = 0
    rejected_total = 0
    previous_selected_camera_id: str | None = None
    previous_selected_observation: MapObservation | None = None
    camera_manager = CameraManager(config.camera_manager_config)
    health_monitors: dict[str, InnovationHealthMonitor] = {}
    health_debouncers: dict[str, HealthDebouncer] = {}

    for idx, frame in enumerate(ordered):
        if idx > 0:
            dx = frame.odometry_xy_m[0] - previous_odom[0]
            dy = frame.odometry_xy_m[1] - previous_odom[1]
            mean = (mean[0] + dx, mean[1] + dy)
            cov = _mat_add(cov, _matrix(config.process_noise_m2, "process_noise_m2"))
            previous_odom = frame.odometry_xy_m

        handover_diagnostics: tuple[HandoverUncertaintyDiagnostic, ...] = tuple()
        selected_handover_observation: MapObservation | None = None
        camera_manager_decision: dict[str, Any] = {}
        frame_health: dict[str, float] = {}
        frame_health_state: dict[str, str] = {}
        dynamic_occlusion_diagnostics: tuple[dict[str, Any], ...] = tuple()
        if config.mode == ReplayMode.HANDOVER_AWARE_SELECTION:
            replay_obs, handover_diagnostics, selected_handover_observation = _handover_observations_for_config(
                frame.observations,
                config,
                mean_xy=mean,
                timestamp_s=frame.timestamp_s,
                previous_camera_id=previous_selected_camera_id,
                previous_observation=previous_selected_observation,
            )
        elif config.mode == ReplayMode.HYSTERETIC_HANDOVER_SELECTION:
            manager_observations = tuple(
                _with_provider_quality(item, config, mean, frame.timestamp_s)
                for item in frame.observations
            )
            manager_decision = camera_manager.select(
                timestamp_s=frame.timestamp_s,
                observations=manager_observations,
            )
            camera_manager_decision = manager_decision.to_dict()
            selected_handover_observation = manager_decision.selected_observation
            adjusted, diagnostic = handover_adjusted_observation(
                previous_camera_id=previous_selected_camera_id,
                selected_observation=selected_handover_observation,
                candidate_observations=manager_observations,
                previous_observation=previous_selected_observation,
                config=config.handover_config,
            )
            replay_obs = (adjusted,) if adjusted is not None else tuple()
            handover_diagnostics = (diagnostic,)
        elif config.mode == ReplayMode.HEALTH_AWARE_FUSION:
            replay_obs, frame_health, frame_health_state = _health_aware_observations(
                frame.observations,
                mean,
                cov,
                health_monitors,
                health_debouncers,
                config,
            )
        elif config.mode == ReplayMode.DYNAMIC_OCCLUSION_AWARE_FUSION:
            replay_obs, dynamic_occlusion_diagnostics = _dynamic_occlusion_observations(
                frame.observations, frame.dynamic_actors, config,
            )
        else:
            replay_obs = _observations_for_config(
                frame.observations,
                config,
                mean_xy=mean,
                timestamp_s=frame.timestamp_s,
            )
        if replay_obs:
            fusion = sequential_kalman_update_2d(
                mean,
                cov,
                replay_obs,
                nis_gate=config.nis_gate,
            )
            mean = fusion.mean_xy
            cov = fusion.covariance_m2
        else:
            fusion = SequentialFusionResult(
                mean_xy=mean,
                covariance_m2=cov,
                accepted_camera_ids=tuple(),
                rejected_camera_ids=tuple(),
                nis_by_camera={},
            )

        if (
            config.mode in (
                ReplayMode.HANDOVER_AWARE_SELECTION,
                ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
            )
            and selected_handover_observation is not None
            and selected_handover_observation.camera_id in fusion.accepted_camera_ids
        ):
            previous_selected_camera_id = selected_handover_observation.camera_id
            previous_selected_observation = selected_handover_observation

        accepted_total += len(fusion.accepted_camera_ids)
        rejected_total += len(fusion.rejected_camera_ids)
        steps.append(
            ReplayStep(
                timestamp_s=frame.timestamp_s,
                mean_xy_m=mean,
                covariance_m2=cov,
                accepted_camera_ids=fusion.accepted_camera_ids,
                rejected_camera_ids=fusion.rejected_camera_ids,
                nis_by_camera=fusion.nis_by_camera,
                handover_diagnostics=tuple(item.to_dict() for item in handover_diagnostics),
                camera_manager_decision=camera_manager_decision,
                health_by_camera=dict(frame_health),
                health_state_by_camera=dict(frame_health_state),
                dynamic_occlusion_diagnostics=dynamic_occlusion_diagnostics,
            )
        )

    metrics = None
    if evaluation_frames:
        metrics = compute_replay_metrics(
            steps,
            evaluation_frames,
            accepted_total=accepted_total,
            rejected_total=rejected_total,
            max_error_divergence_m=config.max_error_divergence_m,
        )
    return ReplayResult(config=config, steps=tuple(steps), metrics=metrics)


def compute_replay_metrics(
    steps: Sequence[ReplayStep],
    evaluation_frames: Sequence[EvaluationFrame],
    *,
    accepted_total: int = 0,
    rejected_total: int = 0,
    max_error_divergence_m: float = 1.0,
) -> ReplayMetrics:
    eval_by_stamp = {round(frame.timestamp_s, 9): frame for frame in evaluation_frames}
    errors: list[float] = []
    nees: list[float] = []
    nis: list[float] = []
    divergence_count = 0
    handover_count = 0
    unqualified_handover_count = 0
    for step in steps:
        eval_frame = eval_by_stamp.get(round(step.timestamp_s, 9))
        if eval_frame is not None:
            err = (
                step.mean_xy_m[0] - eval_frame.truth_xy_m[0],
                step.mean_xy_m[1] - eval_frame.truth_xy_m[1],
            )
            e_norm = math.hypot(err[0], err[1])
            errors.append(e_norm)
            if e_norm > max_error_divergence_m:
                divergence_count += 1
            try:
                nees.append(_quad_form_inverse_2x2(err, step.covariance_m2))
            except ContractValidationError:
                pass
        nis.extend(v for v in step.nis_by_camera.values() if math.isfinite(v))
        for diagnostic in step.handover_diagnostics:
            if not bool(diagnostic.get("switched", False)):
                continue
            handover_count += 1
            reasons = {str(reason) for reason in diagnostic.get("reasons", [])}
            if reasons.intersection(
                {"overlap_disagreement", "no_overlap_confirmation", "async_handover", "selected_stale"}
            ):
                unqualified_handover_count += 1

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else math.nan
    final_error = errors[-1] if errors else math.nan
    acceptance_den = accepted_total + rejected_total
    return ReplayMetrics(
        rmse_m=rmse,
        p95_error_m=_percentile(errors, 95.0),
        max_error_m=max(errors) if errors else math.nan,
        final_error_m=final_error,
        mean_nis=sum(nis) / len(nis) if nis else math.nan,
        mean_nees=sum(nees) / len(nees) if nees else math.nan,
        covariance_1sigma_coverage=_coverage(nees, 2.30),
        covariance_2sigma_coverage=_coverage(nees, 6.18),
        update_acceptance_rate=(accepted_total / acceptance_den) if acceptance_den else math.nan,
        divergence_count=divergence_count,
        handover_count=handover_count,
        unqualified_handover_count=unqualified_handover_count,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a deterministic linear percentile without a NumPy dependency."""

    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _coverage(nees: Sequence[float], threshold: float) -> float:
    finite = [float(value) for value in nees if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(value <= float(threshold) for value in finite) / len(finite)


def _observations_for_config(
    observations: Sequence[MapObservation],
    config: ReplayConfig,
    *,
    mean_xy: Sequence[float],
    timestamp_s: float,
) -> tuple[MapObservation, ...]:
    if config.mode == ReplayMode.ODOM_ONLY:
        return tuple()
    if config.mode in (ReplayMode.SINGLE_FIXED_R, ReplayMode.SINGLE_FIXED_R_NIS):
        obs = observations[:1]
        return tuple(_with_cov(item, config.fixed_measurement_cov_m2, "fixed_R") for item in obs)
    if config.mode == ReplayMode.DETECTOR_SCORE_R:
        return tuple(_with_detector_score_cov(item, config) for item in observations[:1])
    if config.mode == ReplayMode.CURRENT_GP_R:
        return tuple(_with_provider_quality(item, config, mean_xy, timestamp_s) for item in observations[:1])
    if config.mode == ReplayMode.CONSERVATIVE_SELECTION:
        if not observations:
            return tuple()
        candidates = tuple(
            _with_provider_quality(item, config, mean_xy, timestamp_s) for item in observations
        )
        selected = _select_quality_observation(candidates)
        return (selected,)
    return tuple(observations)


def _health_aware_observations(
    observations: Sequence[MapObservation],
    mean_xy: Sequence[float],
    cov: tuple[tuple[float, float], tuple[float, float]],
    monitors: dict[str, InnovationHealthMonitor],
    debouncers: dict[str, HealthDebouncer],
    config: ReplayConfig,
) -> tuple[tuple[MapObservation, ...], dict[str, float], dict[str, str]]:
    """B6 full method: per-camera innovation-health gating on top of fusion.

    For each observation this frame we compute its innovation and NIS against the
    current (frame-prior) belief, feed a PERSISTENT per-camera
    :class:`InnovationHealthMonitor` + :class:`HealthDebouncer`, and act on the
    debounced state:

    - HEALTHY   -> accept the observation unchanged;
    - SUSPECT / RECOVERING -> inflate its covariance by ``health_suspect_inflation``
      (down-weight without discarding);
    - DEGRADED  -> reject it (drop from the update).

    The debouncer's consistency signal is ``NIS <= gate AND health >= threshold``.
    Using the continuous health (not the raw per-frame NIS gate alone) is what lets
    B6 catch the *gate-evading moderate-drift band*: a persistent moderate bias keeps
    each frame's NIS below the gate yet erodes the NIS/bias EWMA, so health falls,
    the debouncer trips SUSPECT->DEGRADED, and the drifting camera is contained —
    exactly the mechanism WP5 validated single-camera. GROUND TRUTH IS NEVER USED:
    innovation is measurement-minus-prior, all operational.
    """

    health_cfg = config.health_config or InnovationHealthConfig()
    deb_cfg = config.health_debouncer_config or HealthDebouncerConfig()
    gate = config.nis_gate if config.nis_gate is not None else 9.21
    inflation = float(config.health_suspect_inflation)
    threshold = float(config.health_inflate_threshold)

    accepted: list[MapObservation] = []
    health_vals: dict[str, float] = {}
    health_states: dict[str, str] = {}
    for obs in observations:
        cam = obs.camera_id
        monitor = monitors.get(cam)
        if monitor is None:
            monitor = InnovationHealthMonitor(cam, health_cfg)
            monitors[cam] = monitor
        debouncer = debouncers.get(cam)
        if debouncer is None:
            debouncer = HealthDebouncer(deb_cfg)
            debouncers[cam] = debouncer

        innovation = (obs.xy_m[0] - mean_xy[0], obs.xy_m[1] - mean_xy[1])
        innovation_cov = _mat_add(cov, obs.covariance_m2)
        try:
            nis = _quad_form_inverse_2x2(innovation, innovation_cov)
        except ContractValidationError:
            nis = math.inf

        health = monitor.update(
            nis=(nis if math.isfinite(nis) else None),
            innovation_uv=innovation,
            dropped=False,
        )
        consistent = math.isfinite(nis) and nis <= gate and health >= threshold
        state = debouncer.step(consistent)
        health_vals[cam] = health
        health_states[cam] = state.value

        policy = HealthDebouncer.response_policy(state)
        if policy == "reject":
            continue
        if policy in ("inflate", "slow_reentry"):
            inflated = (
                (obs.covariance_m2[0][0] * inflation, obs.covariance_m2[0][1] * inflation),
                (obs.covariance_m2[1][0] * inflation, obs.covariance_m2[1][1] * inflation),
            )
            accepted.append(_with_cov(obs, inflated, "health_inflated"))
        else:
            accepted.append(obs)
    return tuple(accepted), health_vals, health_states


def _dynamic_occlusion_observations(
    observations: Sequence[MapObservation],
    actors: Sequence[DynamicActorState],
    config: ReplayConfig,
) -> tuple[tuple[MapObservation, ...], tuple[dict[str, Any], ...]]:
    """D1: make the camera update conservative while tracked actors cross its ray."""

    if config.dynamic_occlusion_config is None:
        raise ContractValidationError(
            "DYNAMIC_OCCLUSION_AWARE_FUSION requires dynamic_occlusion_config"
        )
    adjusted: list[MapObservation] = []
    diagnostics: list[dict[str, Any]] = []
    for observation in observations:
        item, diagnostic = dynamic_occlusion_adjusted_observation(
            observation, actors, config.dynamic_occlusion_config,
        )
        adjusted.append(item)
        diagnostics.append({
            "camera_id": diagnostic.camera_id,
            "occlusion_probability": diagnostic.occlusion_probability,
            "covariance_inflation": diagnostic.covariance_inflation,
            "blocking_actor_ids": list(diagnostic.blocking_actor_ids),
        })
    return tuple(adjusted), tuple(diagnostics)


def _handover_observations_for_config(
    observations: Sequence[MapObservation],
    config: ReplayConfig,
    *,
    mean_xy: Sequence[float],
    timestamp_s: float,
    previous_camera_id: str | None,
    previous_observation: MapObservation | None,
) -> tuple[tuple[MapObservation, ...], tuple[HandoverUncertaintyDiagnostic, ...], MapObservation | None]:
    if not observations:
        diagnostic = handover_adjusted_observation(
            previous_camera_id=previous_camera_id,
            selected_observation=None,
            candidate_observations=tuple(),
            previous_observation=previous_observation,
            config=config.handover_config,
        )[1]
        return tuple(), (diagnostic,), None
    candidates = tuple(
        _with_provider_quality(item, config, mean_xy, timestamp_s) for item in observations
    )
    selected = _select_quality_observation(candidates)
    adjusted, diagnostic = handover_adjusted_observation(
        previous_camera_id=previous_camera_id,
        selected_observation=selected,
        candidate_observations=candidates,
        previous_observation=previous_observation,
        config=config.handover_config,
    )
    return (adjusted,) if adjusted is not None else tuple(), (diagnostic,), selected


def _select_quality_observation(observations: Sequence[MapObservation]) -> MapObservation:
    return max(
        observations,
        key=lambda item: (
            item.quality.p_available * item.quality.association_confidence,
            -float(item.quality.stale),
            -float(item.timestamp_s),
        ),
    )


def _with_cov(obs: MapObservation, cov, source: str) -> MapObservation:
    return MapObservation(
        camera_id=obs.camera_id,
        timestamp_s=obs.timestamp_s,
        xy_m=obs.xy_m,
        covariance_m2=cov,
        quality=CameraQuality(
            camera_id=obs.camera_id,
            p_available=obs.quality.p_available,
            conditional_cov_uv=obs.quality.conditional_cov_uv,
            association_confidence=obs.quality.association_confidence,
            epistemic_score=obs.quality.epistemic_score,
            stale=obs.quality.stale,
            source_model=source,
        ),
        source=source,
    )


def _with_detector_score_cov(obs: MapObservation, config: ReplayConfig) -> MapObservation:
    score = max(min(float(obs.quality.p_available), 1.0), 0.0)
    return _with_probability_cov(obs, config, score, "detector_score_R", obs.quality)


def _with_provider_quality(
    obs: MapObservation,
    config: ReplayConfig,
    mean_xy: Sequence[float],
    timestamp_s: float,
) -> MapObservation:
    provider = dict(config.quality_providers or {}).get(obs.camera_id)
    if provider is None:
        return obs
    quality = provider.query(obs.camera_id, mean_xy, timestamp_s)
    return _with_probability_cov(obs, config, quality.p_available, quality.source_model, quality)


def _with_probability_cov(
    obs: MapObservation,
    config: ReplayConfig,
    score: float,
    source: str,
    quality: CameraQuality,
) -> MapObservation:
    score = max(min(float(score), 1.0), 0.0)
    var = (
        score * float(config.detector_score_min_cov_m2)
        + (1.0 - score) * float(config.detector_score_max_cov_m2)
    )
    return MapObservation(
        camera_id=obs.camera_id,
        timestamp_s=obs.timestamp_s,
        xy_m=obs.xy_m,
        covariance_m2=((var, 0.0), (0.0, var)),
        quality=quality,
        source=source,
    )


def _finite(value: float, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(out):
        raise ContractValidationError(f"{field_name} must be finite")
    return out


def _pair(value: Sequence[float], field_name: str) -> tuple[float, float]:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2-element sequence")
    return (_finite(value[0], f"{field_name}[0]"), _finite(value[1], f"{field_name}[1]"))


def _matrix(value: Sequence[Sequence[float]], field_name: str):
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 2:
            raise ContractValidationError(f"{field_name} must be a 2x2 matrix")
        rows.append((_finite(row[0], f"{field_name}[{i}][0]"), _finite(row[1], f"{field_name}[{i}][1]")))
    return (rows[0], rows[1])


def _mat_add(a, b):
    return ((a[0][0] + b[0][0], a[0][1] + b[0][1]), (a[1][0] + b[1][0], a[1][1] + b[1][1]))


def _mat_inv_2x2(a):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) <= 1.0e-12:
        raise ContractValidationError("matrix is singular")
    inv_det = 1.0 / det
    return ((a[1][1] * inv_det, -a[0][1] * inv_det), (-a[1][0] * inv_det, a[0][0] * inv_det))


def _mat_vec(a, v):
    return (a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1])


def _quad_form_inverse_2x2(v, a):
    inv = _mat_inv_2x2(a)
    iv = _mat_vec(inv, v)
    return float(v[0] * iv[0] + v[1] * iv[1])
