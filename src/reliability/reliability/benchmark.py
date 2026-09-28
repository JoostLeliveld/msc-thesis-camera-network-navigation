"""Replay benchmark suites for reliability and fusion baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from reliability.providers import CameraReliabilityProvider
from reliability.replay import (
    EvaluationFrame,
    ReplayConfig,
    ReplayFrame,
    ReplayMode,
    ReplayResult,
    required_replay_configs,
    run_replay,
)


@dataclass(frozen=True)
class ReplayBenchmarkCondition:
    name: str
    config: ReplayConfig


@dataclass(frozen=True)
class ReplayBenchmarkResult:
    condition: ReplayBenchmarkCondition
    result: ReplayResult


@dataclass(frozen=True)
class ReplayBenchmarkSuite:
    results: tuple[ReplayBenchmarkResult, ...]

    def summary(self) -> dict[str, Any]:
        return summarize_benchmark_results(self.results)


def default_replay_benchmark_conditions(
    *,
    include_multicamera: bool = False,
    quality_providers: Mapping[str, CameraReliabilityProvider] | None = None,
) -> tuple[ReplayBenchmarkCondition, ...]:
    """Return the required replay conditions plus optional fusion baselines."""

    conditions = [
        ReplayBenchmarkCondition(name=config.mode.value, config=config)
        for config in required_replay_configs(quality_providers=quality_providers)
    ]
    if include_multicamera:
        providers = dict(quality_providers or {})
        conditions.extend(
            [
                ReplayBenchmarkCondition(
                    name=ReplayMode.SEQUENTIAL_FUSION.value,
                    config=ReplayConfig(mode=ReplayMode.SEQUENTIAL_FUSION, nis_gate=9.21),
                ),
                ReplayBenchmarkCondition(
                    name=ReplayMode.CONSERVATIVE_SELECTION.value,
                    config=ReplayConfig(
                        mode=ReplayMode.CONSERVATIVE_SELECTION,
                        nis_gate=9.21,
                        quality_providers=providers,
                    ),
                ),
                ReplayBenchmarkCondition(
                    name=ReplayMode.HANDOVER_AWARE_SELECTION.value,
                    config=ReplayConfig(
                        mode=ReplayMode.HANDOVER_AWARE_SELECTION,
                        nis_gate=9.21,
                        quality_providers=providers,
                    ),
                ),
                ReplayBenchmarkCondition(
                    name=ReplayMode.HYSTERETIC_HANDOVER_SELECTION.value,
                    config=ReplayConfig(
                        mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
                        nis_gate=9.21,
                        quality_providers=providers,
                    ),
                ),
            ]
        )
    return tuple(conditions)


def run_replay_benchmark(
    frames: Sequence[ReplayFrame],
    *,
    evaluation_frames: Sequence[EvaluationFrame] | None = None,
    include_multicamera: bool | None = None,
    quality_providers: Mapping[str, CameraReliabilityProvider] | None = None,
) -> ReplayBenchmarkSuite:
    """Run a deterministic offline replay suite over shared input frames."""

    if include_multicamera is None:
        include_multicamera = has_multicamera_observations(frames)
    results = []
    for condition in default_replay_benchmark_conditions(
        include_multicamera=include_multicamera,
        quality_providers=quality_providers,
    ):
        results.append(
            ReplayBenchmarkResult(
                condition=condition,
                result=run_replay(
                    frames,
                    condition.config,
                    evaluation_frames=evaluation_frames,
                ),
            )
        )
    return ReplayBenchmarkSuite(results=tuple(results))


def has_multicamera_observations(frames: Sequence[ReplayFrame]) -> bool:
    """Return true when at least one timestamp has observations from 2+ cameras."""

    for frame in frames:
        ids = {obs.camera_id for obs in frame.observations}
        if len(ids) >= 2:
            return True
    return False


def summarize_benchmark_results(results: Sequence[ReplayBenchmarkResult]) -> dict[str, Any]:
    """Build a JSON-safe summary for CLI output and experiment reports."""

    summary: dict[str, Any] = {"results": {}}
    for item in results:
        metrics = item.result.metrics
        summary["results"][item.condition.name] = {
            "mode": item.condition.config.mode.value,
            "steps": len(item.result.steps),
            "rmse_m": None if metrics is None else _none_nan(metrics.rmse_m),
            "p95_error_m": None if metrics is None else _none_nan(metrics.p95_error_m),
            "max_error_m": None if metrics is None else _none_nan(metrics.max_error_m),
            "final_error_m": None if metrics is None else _none_nan(metrics.final_error_m),
            "mean_nis": None if metrics is None else _none_nan(metrics.mean_nis),
            "mean_nees": None if metrics is None else _none_nan(metrics.mean_nees),
            "covariance_1sigma_coverage": None if metrics is None else _none_nan(metrics.covariance_1sigma_coverage),
            "covariance_2sigma_coverage": None if metrics is None else _none_nan(metrics.covariance_2sigma_coverage),
            "update_acceptance_rate": None if metrics is None else _none_nan(metrics.update_acceptance_rate),
            "divergence_count": None if metrics is None else metrics.divergence_count,
            "handover_count": None if metrics is None else metrics.handover_count,
            "unqualified_handover_count": None if metrics is None else metrics.unqualified_handover_count,
        }
    return summary


def _none_nan(value: float) -> float | None:
    return None if isinstance(value, float) and math.isnan(value) else float(value)
