"""Forward-only runtime for the commissioned box-MLP visibility residual."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from reliability.visibility_residual_net import VisibilityPatchResidualNet


GRID_SIZE = 16
SUPPORTED_SCHEMAS = {
    "commissioned_visibility_sensor_model.v1",  # archived ray-frame packages
    "commissioned_visibility_sensor_model.v2",  # canonical ray-frame R0--R2
}
CURRENT_R_MODELS = {
    "R0_global_full", "R1_per_camera_full", "R2_spatial_residual",
    "R3_hierarchical_predictive",
}
PROJECTION_R_MODEL = "Rproj_homography_pixel"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_verified(entry: dict, *, label: str) -> tuple[Path, bytes]:
    path = Path(str(entry.get("path", ""))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = path.read_bytes()
    if _sha256_bytes(data) != str(entry.get("sha256", "")):
        raise ValueError(f"{label} hash differs from the commissioned manifest")
    return path, data


MEAN_MODEL_NAMES = ("box_mlp_visibility_residual", "M4_visibility_patch_residual")


class CommissionedVisibilitySensorModel:
    """Hash-bound visibility-residual correction and its matched covariance."""

    @property
    def requires_capture_heading(self) -> bool:
        """Whether the selected covariance model is conditioned on robot yaw.

        The canonical R0--R2 models are expressed in the camera-to-observation
        ray frame and therefore do not consume an estimator heading.  Only the
        archived spatial covariance model is heading-conditioned.
        """
        return (
            getattr(self, "projection_sigma_px", None) is None
            and getattr(self, "ray_r_samples", None) is None
            and getattr(self, "current_r_samples", None) is None
        )

    def __init__(self, manifest_path: str | Path, *, expected_sha256: str | None = None) -> None:
        path = Path(manifest_path).expanduser()
        data = path.read_bytes()
        self.sha256 = _sha256_bytes(data)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("commissioned visibility manifest hash differs from expected identity")
        manifest = json.loads(data)
        if manifest.get("schema") not in SUPPORTED_SCHEMAS:
            raise ValueError("unsupported commissioned visibility schema")
        status = manifest.get("status")
        if status not in {
            "frozen_before_audit",
            "provisional_train12_navigation_evaluation_pending",
        } or manifest.get("audit_accessed") is not False:
            raise ValueError("commissioned visibility model has an unsupported evidence status")
        # Two registry spellings name the same model: the covariance fit writes
        # "box_mlp_visibility_residual", the 5 Hz bundle rewrites it as
        # "M4_visibility_patch_residual". Both are the box-MLP base plus the
        # gated patch residual, and the file hashes below pin the identity.
        if manifest.get("mean_model") not in MEAN_MODEL_NAMES:
            raise ValueError("runtime requires the selected visibility-residual mean model")
        runtime_covariance_model = str(manifest.get("runtime_covariance_model", ""))
        if runtime_covariance_model not in {
                "R4_image_conditioned_scale", PROJECTION_R_MODEL, *CURRENT_R_MODELS}:
            raise ValueError("runtime requires a supported visibility-residual matched covariance")
        self.runtime_covariance_model = runtime_covariance_model
        self.manifest = manifest
        self.path = path
        self.camera_order = tuple(str(value) for value in manifest["camera_order"])
        self.camera_xy = {
            str(key): np.asarray(value, dtype=float)
            for key, value in manifest["camera_xy_m"].items()
        }
        if set(self.camera_order) != set(self.camera_xy):
            raise ValueError("camera registry and geometry differ")
        self.height, self.width = map(int, manifest["image_shape_hw"])

        _, base_data = _read_verified(manifest["correction_base"], label="box MLP base model")
        self.base_model = joblib.load(io.BytesIO(base_data))
        _, patch_data = _read_verified(manifest["correction_patch"], label="visibility residual model")
        try:
            checkpoint = torch.load(io.BytesIO(patch_data), map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            checkpoint = torch.load(io.BytesIO(patch_data), map_location="cpu")
        checkpoint_schema = checkpoint.get("schema")
        if checkpoint_schema not in {
            "visibility_patch_residual.v1", "visibility_residual_15feature_12_6.v1",
        } or (checkpoint_schema == "visibility_patch_residual.v1"
              and checkpoint.get("grid_size") != GRID_SIZE):
            raise ValueError("visibility-residual checkpoint feature contract is unsupported")
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_sd = np.asarray(checkpoint["feature_sd"], dtype=np.float32)
        if self.feature_mean.shape != self.feature_sd.shape:
            raise ValueError("visibility-residual feature normalization shape mismatch")
        self.patch_model = VisibilityPatchResidualNet(len(self.feature_mean))
        self.patch_model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.patch_model.eval()

        self.current_r_samples = None
        self.ray_r_samples = None
        self.ray_r_prior_covariance = None
        self.current_r_covariance_multiplier = 1.0
        self.projection_sigma_px = None
        if (self.runtime_covariance_model == PROJECTION_R_MODEL
                and manifest.get("schema") == "commissioned_visibility_sensor_model.v2"):
            _, model_data = _read_verified(
                manifest["covariance_models"], label="projection-baseline parameters")
            with np.load(io.BytesIO(model_data), allow_pickle=False) as archive:
                self.projection_sigma_px = float(np.asarray(
                    archive["homography_baseline_sigma_px"]).reshape(-1)[0])
            if not math.isfinite(self.projection_sigma_px) or self.projection_sigma_px <= 0.0:
                raise ValueError("projection baseline requires a positive pixel-noise scale")
            return
        if (self.runtime_covariance_model in CURRENT_R_MODELS
                and manifest.get("schema") == "commissioned_visibility_sensor_model.v2"):
            if self.runtime_covariance_model == "R3_hierarchical_predictive":
                raise ValueError("v2 supports only the canonical R0--R2 ladder")
            _, model_data = _read_verified(
                manifest["covariance_models"], label="ray-frame covariance models"
            )
            with np.load(io.BytesIO(model_data), allow_pickle=False) as archive:
                order = tuple(np.asarray(archive["camera_order"]).astype(str))
                if order != self.camera_order:
                    raise ValueError("covariance-model camera order differs from manifest")
                self.ray_r_global = self._scatter_from_matrix(
                    np.asarray(archive["global_covariance_ray_m2"], dtype=float))
                per_camera = np.asarray(archive["per_camera_covariance_ray_m2"], dtype=float)
                position = np.asarray(archive["spatial_reference_xy_m"], dtype=float)
                moment = np.asarray(archive["spatial_second_moment_ray_m2"], dtype=float)
                camera = np.asarray(archive["spatial_camera"]).astype(str)
                if (per_camera.shape != (len(order), 2, 2)
                        or position.shape != (len(moment), 2)
                        or moment.shape[1:] != (2, 2)
                        or camera.shape != (len(moment),)):
                    raise ValueError("ray-frame covariance artifact has incompatible arrays")
                self.ray_r_per_camera = {
                    camera_id: self._scatter_from_matrix(per_camera[index])
                    for index, camera_id in enumerate(order)
                }
                self.ray_r_samples = {
                    camera_id: {
                        "xy": position[camera == camera_id],
                        "moment": moment[camera == camera_id],
                    }
                    for camera_id in order
                }
                self.current_r_neighbors = int(np.asarray(archive["k_neighbors"]).reshape(-1)[0])
                self.current_r_bandwidth_m = float(
                    np.asarray(archive["length_scale_m"]).reshape(-1)[0])
                if "prior_strength" in archive.files:
                    self.current_r_shrinkage = float(
                        np.asarray(archive["prior_strength"]).reshape(-1)[0])
                    self.ray_r_prior_covariance = self._scatter_from_matrix(
                        np.asarray(archive["prior_covariance_ray_m2"], dtype=float))
                else:
                    self.current_r_shrinkage = float(
                        np.asarray(archive["shrinkage"]).reshape(-1)[0])
            if any(len(value["xy"]) == 0 for value in self.ray_r_samples.values()):
                raise ValueError("ray-frame covariance artifact lacks a camera population")
            if (self.current_r_neighbors < 1 or self.current_r_bandwidth_m <= 0.0
                    or self.current_r_shrinkage < 0.0):
                raise ValueError("invalid ray-frame spatial covariance parameters")
            return

        if self.runtime_covariance_model in CURRENT_R_MODELS:
            _, residual_data = _read_verified(
                manifest["residual_population"], label="selected-correction residual population"
            )
            with np.load(io.BytesIO(residual_data), allow_pickle=False) as archive:
                xy = np.asarray(archive["raw_world"], dtype=float)
                residual = np.asarray(archive["residual_ray"], dtype=float)
                camera = np.asarray(archive["camera"]).astype(str)
                drive = np.asarray(archive["drive"]).astype(str)
            self.current_r_samples = {}
            pooled = np.concatenate([
                residual[camera == camera_id] for camera_id in self.camera_order
            ])
            self.current_r_global = self._scatter(pooled)
            for camera_id in self.camera_order:
                use = camera == camera_id
                if not np.any(use):
                    raise ValueError(f"residual population has no samples for {camera_id}")
                self.current_r_samples[camera_id] = {
                    "xy": xy[use], "residual": residual[use], "drive": drive[use],
                    "global": self._scatter(residual[use]),
                }
            self.current_r_neighbors = int(manifest.get("neighbors", 120))
            self.current_r_bandwidth_m = float(manifest.get("bandwidth_m", 0.75))
            self.current_r_covariance_multiplier = float(
                manifest.get("runtime_covariance_multiplier", 1.0)
            )
            if (self.current_r_neighbors < 1 or self.current_r_bandwidth_m <= 0.0
                    or self.current_r_covariance_multiplier <= 0.0):
                raise ValueError("invalid current matched-R runtime parameters")
            return

        _, parameter_data = _read_verified(manifest["parameters"], label="covariance parameters")
        with np.load(io.BytesIO(parameter_data), allow_pickle=False) as archive:
            for name in archive.files:
                setattr(self, name, np.asarray(archive[name]))
        if self.spatial_base.shape != (len(self.camera_order), 2, 2):
            raise ValueError("spatial base covariance has the wrong shape")
        self.spatial_spec = manifest["spatial"]
        self.image_spec = manifest["image_scale"]

    @staticmethod
    def _scatter(values: np.ndarray) -> np.ndarray:
        covariance = values.T @ values / max(len(values), 1)
        eig, vec = np.linalg.eigh(0.5 * (covariance + covariance.T))
        return (vec * np.maximum(eig, 1.0e-6)) @ vec.T

    def _current_neighbors(self, camera_id: str, point: np.ndarray):
        sample = self.current_r_samples[camera_id]
        distance2 = np.sum((sample["xy"] - point) ** 2, axis=1)
        count = min(self.current_r_neighbors, len(distance2))
        index = np.argpartition(distance2, count - 1)[:count]
        distance = np.sqrt(distance2[index])
        weight = np.exp(-0.5 * (distance / self.current_r_bandwidth_m) ** 2)
        if float(weight.sum()) <= 1.0e-12:
            weight[np.argmin(distance)] = 1.0
        return sample, index, weight

    def _current_r_covariance_ray(self, camera_id: str, point: np.ndarray) -> np.ndarray:
        if self.runtime_covariance_model == "R0_global_full":
            return self.current_r_global
        sample = self.current_r_samples[camera_id]
        if self.runtime_covariance_model == "R1_per_camera_full":
            return sample["global"]
        sample, index, weight = self._current_neighbors(camera_id, point)
        residual = sample["residual"][index]
        drives = sample["drive"][index]
        if self.runtime_covariance_model == "R2_spatial_residual":
            capped = weight.copy()
            for drive in np.unique(drives):
                use = drives == drive
                total = float(capped[use].sum())
                if total > 0.0:
                    capped[use] /= total
            scatter = sum(w * np.outer(value, value) for w, value in zip(capped, residual))
            return self._scatter_from_matrix(
                (scatter + 4.0 * sample["global"]) / (float(capped.sum()) + 4.0)
            )

        means, covariances, drive_weights = [], [], []
        for drive in np.unique(drives):
            use = drives == drive
            local_weight = weight[use]
            if float(local_weight.sum()) <= 1.0e-12:
                continue
            values = residual[use]
            mean = np.average(values, axis=0, weights=local_weight)
            covariance = sum(
                w * np.outer(value - mean, value - mean)
                for w, value in zip(local_weight, values)
            ) / float(local_weight.sum())
            means.append(mean)
            covariances.append(self._scatter_from_matrix(covariance))
            drive_weights.append(min(1.0, float(local_weight.sum()) / 5.0))
        wd = np.asarray(drive_weights, dtype=float)
        n = float(wd.sum())
        if n <= 1.0e-12:
            return sample["global"]
        means = np.asarray(means, dtype=float)
        mean = np.average(means, axis=0, weights=wd)
        within = np.average(np.asarray(covariances), axis=0, weights=wd)
        between = sum(
            w * np.outer(value - mean, value - mean) for w, value in zip(wd, means)
        ) / n
        kappa, nu = 1.0 + n, 5.0 + n
        psi = 2.0 * sample["global"] + n * (within + between) + (n / kappa) * np.outer(mean, mean)
        degrees = max(nu - 1.0, 3.01)
        scale = self._scatter_from_matrix(((kappa + 1.0) / (kappa * degrees)) * psi)
        return self._scatter_from_matrix(scale * degrees / (degrees - 2.0))

    def _ray_r_covariance(self, camera_id: str, point: np.ndarray) -> np.ndarray:
        """Canonical R0--R2 covariance in the camera-to-query ray frame."""
        if self.runtime_covariance_model == "R0_global_full":
            return self.ray_r_global
        if self.runtime_covariance_model == "R1_per_camera_full":
            return self.ray_r_per_camera[camera_id]
        sample = self.ray_r_samples[camera_id]
        distance2 = np.sum((sample["xy"] - point) ** 2, axis=1)
        count = min(self.current_r_neighbors, len(distance2))
        index = np.argpartition(distance2, count - 1)[:count]
        weight = np.exp(-0.5 * distance2[index] / self.current_r_bandwidth_m ** 2)
        numerator = np.einsum("n,nij->ij", weight, sample["moment"][index])
        denominator = float(weight.sum())
        prior = self.current_r_shrinkage
        prior_covariance = (
            self.ray_r_per_camera[camera_id]
            if self.ray_r_prior_covariance is None
            else self.ray_r_prior_covariance
        )
        return self._scatter_from_matrix(
            (numerator + prior * prior_covariance)
            / (denominator + prior)
        )

    @staticmethod
    def _scatter_from_matrix(covariance: np.ndarray) -> np.ndarray:
        eig, vec = np.linalg.eigh(0.5 * (covariance + covariance.T))
        return (vec * np.maximum(eig, 1.0e-6)) @ vec.T

    def _features(
        self, camera_id: str, raw_xy: Sequence[float], bbox_xyxy: Sequence[float], confidence: float
    ) -> np.ndarray:
        if camera_id not in self.camera_xy:
            raise ValueError("camera is outside the commissioned registry")
        raw = np.asarray(raw_xy, dtype=float)
        box = np.asarray(bbox_xyxy, dtype=float)
        if raw.shape != (2,) or box.shape != (4,) or not np.isfinite(np.r_[raw, box, confidence]).all():
            raise ValueError("runtime correction inputs are malformed")
        dx, dy = raw - self.camera_xy[camera_id]
        distance = math.hypot(float(dx), float(dy))
        if distance <= 1e-9 or box[2] <= box[0] or box[3] <= box[1] or not 0.0 <= confidence <= 1.0:
            raise ValueError("runtime correction inputs are outside support")
        width_fraction = (box[2] - box[0]) / self.width
        height_fraction = (box[3] - box[1]) / self.height
        values = [
            distance, 1.0 / distance, math.sin(math.atan2(dy, dx)), math.cos(math.atan2(dy, dx)),
            width_fraction, height_fraction,
            (box[2] - box[0]) / max(box[3] - box[1], 1e-9),
            0.5 * (box[0] + box[2]) / self.width, box[3] / self.height, confidence,
            *[float(camera_id == value) for value in self.camera_order],
        ]
        return np.asarray(values, dtype=np.float32)

    def correction_ray(
        self, camera_id: str, raw_xy: Sequence[float], bbox_xyxy: Sequence[float],
        confidence: float, visibility_grid: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        features = self._features(camera_id, raw_xy, bbox_xyxy, confidence)
        grid = np.asarray(visibility_grid, dtype=np.float32)
        if grid.size != GRID_SIZE * GRID_SIZE:
            raise ValueError("visibility grid must contain exactly 256 values")
        grid = grid.reshape(1, GRID_SIZE, GRID_SIZE)
        if not np.isfinite(grid).all() or grid.min() < 0.0 or grid.max() > 1.0:
            raise ValueError("visibility grid must be finite and in [0,1]")
        base = np.asarray(self.base_model.predict(features[None])[0], dtype=np.float32)
        with torch.inference_mode():
            prediction = self.patch_model(
                torch.from_numpy(grid[None]),
                torch.from_numpy(((features - self.feature_mean) / self.feature_sd)[None]),
                torch.from_numpy(base[None]),
            )[0][0].numpy()  # (corrected, residual, gate) -> first row of corrected
        if prediction.shape != (2,) or not np.isfinite(prediction).all():
            raise ValueError("visibility-residual model produced an invalid correction")
        return prediction.astype(float)

    def _spatial_covariance(self, camera_id: str, x: float, y: float, heading: float) -> np.ndarray:
        selected = self.spatial_camera == camera_id
        train = self.spatial_features[selected]
        residual = self.spatial_residual[selected]
        weight = self.spatial_sample_weight[selected]
        distance2 = np.sum((train[:, :2] - np.asarray([x, y])) ** 2, axis=1)
        heading_term = 1.0 - np.cos(float(heading) - train[:, 2])
        local = np.exp(
            -0.5 * distance2 / float(self.spatial_spec["position_scale_m"]) ** 2
            - heading_term / float(self.spatial_spec["heading_scale"]) ** 2
        ) * weight
        camera_index = self.camera_order.index(camera_id)
        prior = float(self.spatial_spec["prior_strength"])
        numerator = np.einsum("n,ni,nj->ij", local, residual, residual)
        covariance = (numerator + prior * self.spatial_base[camera_index]) / (local.sum() + prior)
        return 0.5 * (covariance + covariance.T) + np.eye(2) * 1.0e-6

    def planner_covariance(
        self, camera_id: str, x: float, y: float, heading: float | None = None
    ) -> np.ndarray:
        if getattr(self, "projection_sigma_px", None) is not None:
            raise ValueError("projection baseline is replay-only and has no planning field")
        if self.ray_r_samples is not None:
            point = np.asarray([x, y], dtype=float)
            ray = point - self.camera_xy[camera_id]
            ray /= np.linalg.norm(ray)
            basis = np.column_stack((ray, np.asarray([-ray[1], ray[0]])))
            covariance_ray = self._ray_r_covariance(camera_id, point)
            return basis @ covariance_ray @ basis.T
        if self.current_r_samples is not None:
            point = np.asarray([x, y], dtype=float)
            ray = point - self.camera_xy[camera_id]
            ray /= np.linalg.norm(ray)
            basis = np.column_stack((ray, np.asarray([-ray[1], ray[0]])))
            return basis @ self._current_r_covariance_ray(camera_id, point) @ basis.T
        if heading is None:
            raise ValueError("legacy spatial covariance requires robot heading")
        covariance = self._spatial_covariance(camera_id, x, y, heading)
        covariance *= float(self.spatial_spec["external_calibration_scale"])
        return covariance

    def correct_and_covariance(
        self, camera_id: str, raw_xy: Sequence[float], bbox_xyxy: Sequence[float],
        confidence: float, visibility_grid: Sequence[float] | np.ndarray,
        heading: float | None = None,
        projection_covariance_world: Sequence[Sequence[float]] | None = None,
    ) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]]:
        correction = self.correction_ray(camera_id, raw_xy, bbox_xyxy, confidence, visibility_grid)
        raw = np.asarray(raw_xy, dtype=float)
        ray = raw - self.camera_xy[camera_id]
        along = ray / np.linalg.norm(ray)
        basis = np.column_stack((along, np.asarray([-along[1], along[0]])))
        corrected = raw + basis @ correction
        if getattr(self, "projection_sigma_px", None) is not None:
            if projection_covariance_world is None:
                raise ValueError("projection baseline requires unit-pixel projected covariance")
            covariance_world = self._scatter_from_matrix(
                np.asarray(projection_covariance_world, dtype=float)
                * self.projection_sigma_px ** 2)
            return (
                (float(corrected[0]), float(corrected[1])),
                ((float(covariance_world[0, 0]), float(covariance_world[0, 1])),
                 (float(covariance_world[1, 0]), float(covariance_world[1, 1]))),
            )
        if self.ray_r_samples is not None:
            covariance_along = corrected - self.camera_xy[camera_id]
            covariance_along /= np.linalg.norm(covariance_along)
            covariance_basis = np.column_stack((
                covariance_along,
                np.asarray([-covariance_along[1], covariance_along[0]]),
            ))
            covariance_ray = self._ray_r_covariance(camera_id, corrected)
            covariance_world = covariance_basis @ covariance_ray @ covariance_basis.T
            return (
                (float(corrected[0]), float(corrected[1])),
                ((float(covariance_world[0, 0]), float(covariance_world[0, 1])),
                 (float(covariance_world[1, 0]), float(covariance_world[1, 1]))),
            )
        if self.current_r_samples is not None:
            covariance_ray = self._current_r_covariance_ray(camera_id, corrected)
            covariance_world = basis @ covariance_ray @ basis.T
            covariance_world *= self.current_r_covariance_multiplier
            covariance_world = 0.5 * (covariance_world + covariance_world.T)
            return (
                (float(corrected[0]), float(corrected[1])),
                ((float(covariance_world[0, 0]), float(covariance_world[0, 1])),
                 (float(covariance_world[1, 0]), float(covariance_world[1, 1]))),
            )
        if heading is None:
            raise ValueError("legacy spatial covariance requires capture-time robot heading")
        covariance_ray = self._spatial_covariance(camera_id, corrected[0], corrected[1], heading)
        grid = np.asarray(visibility_grid, dtype=float).reshape(1, GRID_SIZE, GRID_SIZE)
        visibility_score = 0.5 * float(grid.mean()) + 0.5 * float(grid[:, 3 * GRID_SIZE // 4 :, :].mean())
        features = self._features(camera_id, raw_xy, bbox_xyxy, confidence)
        image_row = np.asarray([
            visibility_score, features[4], features[5], confidence, features[0],
            features[2], features[3], *features[10:],
        ], dtype=float)
        standardized = (image_row - self.image_feature_mean) / self.image_feature_scale
        raw_log_scale = float(np.dot(self.image_ridge_coef, standardized) + self.image_ridge_intercept)
        log_scale = raw_log_scale + float(self.image_spec["internal_log_calibration"])
        scale = float(np.exp(np.clip(
            log_scale,
            math.log(float(self.image_spec["minimum"])),
            math.log(float(self.image_spec["maximum"])),
        )))
        covariance_ray *= scale * float(self.image_spec["external_calibration_scale"])
        covariance_world = basis @ covariance_ray @ basis.T
        covariance_world = 0.5 * (covariance_world + covariance_world.T)
        return (
            (float(corrected[0]), float(corrected[1])),
            ((float(covariance_world[0, 0]), float(covariance_world[0, 1])),
             (float(covariance_world[1, 0]), float(covariance_world[1, 1]))),
        )
