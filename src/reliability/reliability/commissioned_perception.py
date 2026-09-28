"""Runtime-only commissioned perception mean and covariance models.

The public boundary is a corrected world-XY observation and a matched 2x2
covariance.  Inputs are restricted to the current detector frame, raw ground
projection, box geometry, confidence, and camera identity.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np


CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
FEATURE_NAMES = (
    "raw_x_m", "raw_y_m", "raw_range_m", "inverse_raw_range",
    "ray_bearing_cos", "ray_bearing_sin",
    "bbox_width_fraction", "bbox_height_fraction", "bbox_aspect",
    "bbox_bottom_u_fraction", "bbox_bottom_v_fraction", "confidence",
    *(f"is_{camera}" for camera in CAMERAS),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spd(value: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    matrix = 0.5 * (np.asarray(value, dtype=float) + np.asarray(value, dtype=float).T)
    eig, vec = np.linalg.eigh(matrix)
    return (vec * np.maximum(eig, floor)) @ vec.T


def _ray_basis(camera_xy: np.ndarray, raw_xy: np.ndarray) -> np.ndarray:
    along = raw_xy - camera_xy
    distance = float(np.linalg.norm(along))
    if not math.isfinite(distance) or distance <= 1e-9:
        raise ValueError("invalid camera-to-observation ray")
    along /= distance
    return np.column_stack((along, np.asarray((-along[1], along[0]))))


class _BoxSpatialMLP:
    @staticmethod
    def build(torch, inputs: int):
        nn = torch.nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(inputs, 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU(),
                    nn.Linear(64, 2),
                )

            def forward(self, feature):
                return self.layers(feature)

        return Network()


class _RGBGaussianNet:
    @staticmethod
    def build(torch, feature_dim: int):
        nn = torch.nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(4, 24, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(48, 72, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(72, 96, 3, 2, 1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                )
                self.shared = nn.Sequential(
                    nn.Linear(feature_dim + 96, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(),
                )
                self.mean_head = nn.Linear(64, 2)
                self.cholesky_head = nn.Linear(64, 3)

            def forward(self, image, feature):
                hidden = self.shared(torch.cat((self.encoder(image), feature), dim=1))
                mean = self.mean_head(hidden)
                raw = self.cholesky_head(hidden)
                d1 = torch.nn.functional.softplus(raw[:, 0]) + 1e-3
                d2 = torch.nn.functional.softplus(raw[:, 2]) + 1e-3
                zeros = torch.zeros_like(d1)
                factor = torch.stack((
                    torch.stack((d1, zeros), dim=1),
                    torch.stack((raw[:, 1], d2), dim=1),
                ), dim=1)
                return mean, factor

        return Network()


class CommissionedPerceptionSensorModel:
    """Load one of the three provisional uncertainty-method packages."""

    SCHEMA = "commissioned_perception_runtime_model.v1"
    METHODS = {"global_residual", "per_camera_residual", "hierarchical_residual",
               "spatial_residual", "joint_rgb_gaussian"}

    def __init__(self, package_path: str, *, expected_sha256: str | None = None):
        self.path = Path(package_path).expanduser().resolve()
        self.sha256 = _sha256(self.path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(
                f"perception sensor-model SHA-256 mismatch: expected {expected_sha256}, "
                f"got {self.sha256}"
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != self.SCHEMA:
            raise ValueError("unsupported commissioned perception package schema")
        self.method = str(payload.get("method_id", ""))
        if self.method not in self.METHODS:
            raise ValueError(f"unsupported commissioned perception method {self.method!r}")

        import torch
        self.torch = torch
        checkpoint_path = (self.path.parent / payload["mean_checkpoint"]).resolve()
        expected_checkpoint = str(payload.get("mean_checkpoint_sha256", ""))
        if _sha256(checkpoint_path) != expected_checkpoint:
            raise ValueError("mean checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "perception_world_position_model.v1":
            raise ValueError("unsupported mean checkpoint schema")
        if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("mean checkpoint feature contract mismatch")
        self.model_kind = str(checkpoint.get("model_kind", ""))
        expected_kind = "rgb_gaussian" if self.method == "joint_rgb_gaussian" else "box_spatial_mlp"
        if self.model_kind != expected_kind:
            raise ValueError(f"{self.method} requires {expected_kind}, got {self.model_kind}")
        if self.model_kind == "box_spatial_mlp":
            self.model = _BoxSpatialMLP.build(torch, len(FEATURE_NAMES))
        else:
            self.model = _RGBGaussianNet.build(torch, len(FEATURE_NAMES))
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

        self.samples: dict[str, dict[str, np.ndarray]] = {}
        self.k = int(payload.get("neighbors", 120))
        self.bandwidth_m = float(payload.get("bandwidth_m", 0.75))
        if self.method != "joint_rgb_gaussian":
            residual_path = (self.path.parent / payload["residual_artifact"]).resolve()
            if _sha256(residual_path) != str(payload.get("residual_artifact_sha256", "")):
                raise ValueError("residual artifact hash mismatch")
            with np.load(residual_path, allow_pickle=False) as archive:
                xy = np.asarray(archive["raw_world"], dtype=float)
                residual = np.asarray(archive["residual_ray"], dtype=float)
                camera = np.asarray(archive["camera"]).astype(str)
                drive = np.asarray(archive["drive"]).astype(str)
            for camera_id in CAMERAS:
                use = camera == camera_id
                if not np.any(use):
                    raise ValueError(f"residual artifact has no samples for {camera_id}")
                values = residual[use]
                self.samples[camera_id] = {
                    "xy": xy[use], "residual": values, "drive": drive[use],
                    "global": _spd(values.T @ values / len(values)),
                }

    def _features(self, camera_id, raw_xy, bbox, confidence, camera_xy, image_size):
        if camera_id not in CAMERAS or bbox is None:
            raise ValueError("unsupported camera or missing box")
        raw = np.asarray(raw_xy, dtype=float)
        cam = np.asarray(camera_xy, dtype=float)
        x0, y0, x1, y1 = map(float, bbox)
        width_px, height_px = map(float, image_size)
        bw, bh = x1 - x0, y1 - y0
        basis = _ray_basis(cam, raw)
        ray = raw - cam
        distance = float(np.linalg.norm(ray))
        bearing = math.atan2(ray[1], ray[0])
        feature = np.asarray([
            raw[0], raw[1], distance, 1.0 / max(distance, 1e-6),
            math.cos(bearing), math.sin(bearing), bw / width_px, bh / height_px,
            bw / max(bh, 1e-6), 0.5 * (x0 + x1) / width_px, y1 / height_px,
            float(confidence), *[float(camera_id == value) for value in CAMERAS],
        ], dtype=np.float32)
        return raw, basis, (feature - self.feature_mean) / self.feature_std

    def _neighbors(self, camera_id: str, raw: np.ndarray):
        sample = self.samples[camera_id]
        d2 = np.sum((sample["xy"] - raw) ** 2, axis=1)
        count = min(self.k, len(d2))
        index = np.argpartition(d2, count - 1)[:count]
        distance = np.sqrt(d2[index])
        weight = np.exp(-0.5 * (distance / self.bandwidth_m) ** 2)
        if float(weight.sum()) <= 1e-12:
            weight[np.argmin(distance)] = 1.0
        return sample, index, weight

    def _spatial_covariance(self, camera_id: str, raw: np.ndarray):
        sample, index, weight = self._neighbors(camera_id, raw)
        residual = sample["residual"][index]
        drives = sample["drive"][index]
        for drive in np.unique(drives):
            use = drives == drive
            total = float(weight[use].sum())
            if total > 0.0:
                weight[use] /= total
        scatter = sum(w * np.outer(e, e) for w, e in zip(weight, residual))
        return np.zeros(2), _spd((scatter + 4.0 * sample["global"]) / (weight.sum() + 4.0))

    def _constant_covariance(self, camera_id: str):
        if self.method == "per_camera_residual":
            return np.zeros(2), self.samples[camera_id]["global"]
        values = np.concatenate([sample["residual"] for sample in self.samples.values()])
        return np.zeros(2), _spd(values.T @ values / len(values))

    def _hierarchical_predictive(self, camera_id: str, raw: np.ndarray):
        sample, index, weight = self._neighbors(camera_id, raw)
        residual = sample["residual"][index]
        drives = sample["drive"][index]
        means, covariances, drive_weights = [], [], []
        for drive in np.unique(drives):
            use = drives == drive
            w = weight[use]
            if float(w.sum()) <= 1e-12:
                continue
            values = residual[use]
            mean = np.average(values, axis=0, weights=w)
            covariance = sum(
                wi * np.outer(value - mean, value - mean)
                for wi, value in zip(w, values)
            ) / float(w.sum())
            means.append(mean)
            covariances.append(_spd(covariance))
            drive_weights.append(min(1.0, float(w.sum()) / 5.0))
        wd = np.asarray(drive_weights)
        n = float(wd.sum())
        if n <= 1e-12:
            return np.zeros(2), sample["global"]
        means = np.asarray(means)
        mean = np.average(means, axis=0, weights=wd)
        within = np.average(np.asarray(covariances), axis=0, weights=wd)
        between = sum(w * np.outer(value - mean, value - mean) for w, value in zip(wd, means)) / n
        kappa0, nu0 = 1.0, 5.0
        kappa, nu = kappa0 + n, nu0 + n
        psi0 = sample["global"] * (nu0 - 3.0)
        posterior_mean = n * mean / kappa
        psi = (
            psi0 + n * (within + between)
            + (kappa0 * n / kappa) * np.outer(mean, mean)
        )
        degrees = max(nu - 1.0, 3.01)
        scale = _spd(((kappa + 1.0) / (kappa * degrees)) * psi)
        predictive_covariance = _spd(scale * degrees / (degrees - 2.0))
        return posterior_mean, predictive_covariance

    def correct_and_covariance(
        self, camera_id, raw_xy, bbox, confidence, rgb_crop_b64, camera_position,
        image_size=(1280.0, 720.0),
    ):
        raw, basis, feature = self._features(
            camera_id, raw_xy, bbox, confidence, camera_position[:2], image_size
        )
        torch = self.torch
        feature_tensor = torch.from_numpy(feature[None]).to(dtype=torch.float32)
        with torch.no_grad():
            if self.model_kind == "box_spatial_mlp":
                mean_ray = self.model(feature_tensor).numpy()[0]
                if self.method in {"global_residual", "per_camera_residual"}:
                    residual_mean, covariance_ray = self._constant_covariance(camera_id)
                elif self.method == "hierarchical_residual":
                    residual_mean, covariance_ray = self._hierarchical_predictive(camera_id, raw)
                else:
                    residual_mean, covariance_ray = self._spatial_covariance(camera_id, raw)
                mean_ray = mean_ray + residual_mean
            else:
                if not rgb_crop_b64:
                    raise ValueError("joint RGB Gaussian needs current-frame RGB context crop")
                packed = base64.b64decode(rgb_crop_b64, validate=True)
                crop = np.frombuffer(zlib.decompress(packed), dtype=np.uint8)
                if crop.size != 3 * 96 * 96:
                    raise ValueError("RGB context crop has invalid decompressed size")
                rgb = crop.reshape(3, 96, 96).astype(np.float32) / 255.0
                image = np.concatenate((rgb, np.ones((1, 96, 96), dtype=np.float32)), axis=0)
                mean_tensor, factor_tensor = self.model(
                    torch.from_numpy(image[None]), feature_tensor
                )
                mean_ray = mean_tensor.numpy()[0]
                factor = factor_tensor.numpy()[0]
                covariance_ray = _spd(factor @ factor.T)
        corrected = raw + basis @ np.asarray(mean_ray, dtype=float)
        covariance_world = _spd(basis @ covariance_ray @ basis.T)
        return tuple(map(float, corrected)), tuple(tuple(map(float, row)) for row in covariance_world)
