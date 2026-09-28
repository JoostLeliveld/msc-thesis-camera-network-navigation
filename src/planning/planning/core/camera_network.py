"""Camera-network fields used by the legacy and metric planning objectives.

The current metric interface stores the precision supplied by the covariance
model matched to each experimental arm. Older expected-information, score and
Bernoulli-report schemas remain readable only for historical reproducibility.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import MappingProxyType
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from planning.core.plan_validation import immutable_array, validate_covariance


@lru_cache(maxsize=5)
def _binary_report_masks(camera_count: int) -> np.ndarray:
    """All hit/miss report branches as a reusable numeric matrix."""
    masks = np.asarray(list(product((0, 1), repeat=camera_count)), dtype=float)
    masks.setflags(write=False)
    return masks


def _source_hashes(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('network source_hashes must be a nonempty mapping')
    for path, digest in value.items():
        if (not isinstance(path, str) or not path.strip() or Path(path).is_absolute()
                or '..' in Path(path).parts or not isinstance(digest, str)
                or re.fullmatch(r'[0-9a-f]{64}', digest) is None):
            raise ValueError('network source_hashes requires relative names and SHA-256 digests')
    return dict(value)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _spd(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape[-2:] != (2, 2) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain finite 2x2 matrices')
    if not np.allclose(value, value.swapaxes(-1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f'{name} must be symmetric')
    if np.linalg.eigvalsh(value).min() <= 0:
        raise ValueError(f'{name} must be positive definite')
    return value


def _psd(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape[-2:] != (2, 2) or not np.isfinite(value).all():
        raise ValueError(f'{name} must contain finite 2x2 matrices')
    if not np.allclose(value, value.swapaxes(-1, -2), atol=1e-12, rtol=1e-10):
        raise ValueError(f'{name} must be symmetric')
    scale = np.maximum(1.0, np.max(np.abs(value), axis=(-2, -1)))
    if np.any(np.linalg.eigvalsh(value)[..., 0] < -1e-12 * scale):
        raise ValueError(f'{name} must be positive semidefinite')
    return value


def projection_jacobian(H, state):
    """Original-image pixels per metre for the fixed cost chart, not a detector."""
    H = np.asarray(H, dtype=float)
    state = np.asarray(state, dtype=float)
    if H.shape != (3,3) or not np.isfinite(H).all() or state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError('cost chart requires finite 3x3 homography and [x,y,yaw]')
    if np.linalg.matrix_rank(H) != 3:
        raise ValueError('cost chart homography must be nonsingular')
    point = np.r_[state[:2], 1.]
    projected = H @ point
    if abs(projected[2]) < 1e-8:
        raise ValueError('network cost chart is singular at this query')
    return (H[:2, :2]*projected[2] - projected[:2, None]*H[2, :2])/projected[2]**2


# Shared ambiguity floor: ONE isotropic position covariance, identical for every
# condition. It enters the ambiguity term only as a constant subtraction, so any value
# below the tightest reachable R_eff gives identical rankings; the requirement is
# that it never clips a real pose. Kept here so both the direct and historical
# ambiguity path anchor to the same place. Mirrors
# UnicyclePlannerBase.AMBIGUITY_FLOOR_POSITION_SD_M.
AMBIGUITY_FLOOR_POSITION_SD_M = 1.5e-3

# Fixed information regularizer for the ambiguity-only equivalent covariance.
# Units are m^-2. At zero expected camera information this gives a declared
# 1 m isotropic standard deviation. It is shared by every arm, independent of
# process noise, and never enters belief propagation.
AMBIGUITY_INFORMATION_REGULARIZER_M2_INV = 1.0

DIRECT_INFORMATION_SCHEMAS = (
    'camera_network.thesis_stage09.v3',
    'camera_network.final_bayesian_planning.v1',
    'camera_network.matched_covariance_precision.v1',
)


class CameraNetworkModel:
    """Frozen per-camera planning fields.

    Historical schemas contain availability and conditional covariance. The
    current schema contains expected information with layout
    ``[camera,y,x,2,2]`` and opportunity support ``[camera,y,x]``. Bilinear
    interpolation matches the CasADi path. Outside the commissioned grid all
    fields are zero. No image or ground truth is queried.
    """
    def __setattr__(self, name, value):
        if getattr(self, '_loaded', False) and not name.startswith('_'):
            raise AttributeError('loaded camera network is immutable; construct a new model')
        object.__setattr__(self, name, value)

    def __init__(self, artifact_path, cameras=None, *, expected_sha256=None,
                 expected_source_hashes=None, expected_camera_ids=None):
        self.path = Path(artifact_path).expanduser().resolve()
        artifact_bytes = self.path.read_bytes()
        self.sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('camera-network artifact SHA-256 mismatch')
        # Hash and parse one immutable byte snapshot; a replaced file cannot race
        # a separate np.load(path) into having the old identity and new arrays.
        with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata_json'].item()))
            if not isinstance(self.metadata, dict):
                raise ValueError('network metadata must be an object')
            sources = _source_hashes(self.metadata.get('source_hashes'))
            if expected_source_hashes is not None and sources != _source_hashes(expected_source_hashes):
                raise ValueError('camera-network source provenance differs from expected manifest')
            schema = self.metadata.get('schema')
            if schema not in ('camera_network.iwai.v1', 'camera_network.thesis_stage09.v1',
                              'camera_network.thesis_stage09.v2',
                              *DIRECT_INFORMATION_SCHEMAS):
                raise ValueError('unsupported camera-network artifact schema')
            if self.metadata.get('reference') != 'robot_ground_reference_xy':
                raise ValueError('network must measure the declared ground reference')
            if self.metadata.get('frame') != 'map_bev':
                raise ValueError('network field must use the map_bev frame')
            self.direct_information = schema in DIRECT_INFORMATION_SCHEMAS
            if not self.direct_information and self.metadata.get('covariance_units') != 'm2':
                raise ValueError('network covariance must be square metres')
            if schema == 'camera_network.final_bayesian_planning.v1':
                if self.metadata.get('fit_role') != 'D_R':
                    raise ValueError('final Bayesian planning fields must be fitted on D_R')
                if self.metadata.get('D_eval_accessed') is not False:
                    raise ValueError('final Bayesian planning fields must not access D_eval')
            if schema == 'camera_network.iwai.v1':
                if self.metadata.get('score_target') != 'detector_score_with_miss_zero':
                    raise ValueError('the IWAI proxy requires an explicitly labelled detector-score field')
                if self.metadata.get('availability_target') != 'valid_detection_finite_ground_projection':
                    raise ValueError('availability must describe the declared pre-gate detection event')
            elif not self.direct_information:
                if self.metadata.get('score_target') != 'unused_in_metric_expected_belief':
                    raise ValueError('Stage-09 score must be explicitly marked unused')
                if self.metadata.get('availability_target') != 'stage06_admitted_localization_measurement':
                    raise ValueError('Stage-09 availability must target complete-chain admission')
            raw_ids = np.asarray(data['camera_ids'])
            if raw_ids.ndim != 1 or raw_ids.dtype.kind not in ('U','S'):
                raise ValueError('camera IDs must be a one-dimensional string axis')
            ids = tuple(raw_ids.astype(str))
            if not ids or len(set(ids)) != len(ids) or any(not c or c != c.strip() for c in ids):
                raise ValueError('unique camera IDs required')
            if expected_camera_ids is not None:
                expected = tuple(expected_camera_ids)
                if not expected or len(set(expected)) != len(expected) or set(expected) != set(ids):
                    raise ValueError('camera-network roster differs from declared future cameras')
            self.camera_ids = ids if cameras is None else tuple(cameras)
            if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
                raise ValueError('camera mask must be nonempty and unique')
            if not set(self.camera_ids) <= set(ids):
                raise ValueError('camera mask contains an unknown camera')
            indices = [ids.index(c) for c in self.camera_ids]
            self.xs, self.ys = np.asarray(data['xs'], float), np.asarray(data['ys'], float)
            for axis in (self.xs, self.ys):
                if axis.ndim != 1 or len(axis)<2 or not np.isfinite(axis).all() or not (np.diff(axis)>0).all():
                    raise ValueError('network axes must be finite and strictly increasing')
            self.fields = {}
            self.dynamic_R = False
            if self.direct_information:
                matched_covariance = schema == 'camera_network.matched_covariance_precision.v1'
                expected_target = (
                    'inverse_of_matched_runtime_covariance'
                    if matched_covariance else 'admitted_runtime_precision_else_zero')
                if self.metadata.get('planning_target') != expected_target:
                    raise ValueError('direct-information fields must declare the planning target')
                information_key = (
                    'matched_precision_m2_inv'
                    if matched_covariance else 'expected_information_m2_inv')
                full_information = _psd(data[information_key], 'camera information')
                expected = (len(ids), len(self.ys), len(self.xs), 2, 2)
                if full_information.shape != expected:
                    raise ValueError(
                        f'expected information shape differs: {full_information.shape} != {expected}')
                support_key = 'residual_support' if matched_covariance else 'opportunity_support'
                support = np.asarray(data[support_key], dtype=float)
                if (support.shape != expected[:3] or not np.isfinite(support).all()
                        or np.any(support < 0.0)):
                    raise ValueError('field support must be finite and nonnegative [camera,y,x]')
                self.expected_information = immutable_array(full_information[indices])
                self.fields[
                    'residual_support' if matched_covariance else 'opportunity_support'
                ] = immutable_array(support[indices])
            else:
                for key in ('score', 'availability'):
                    grid = np.asarray(data[key], float)
                    if grid.shape != (len(ids), len(self.ys), len(self.xs)):
                        raise ValueError(f'{key} camera/y/x dimensions differ')
                    if not np.isfinite(grid).all() or np.any((grid<0)|(grid>1)):
                        raise ValueError(f'{key} must be in [0,1]')
                    self.fields[key] = immutable_array(grid[indices])
                full_R = _spd(data['R_cond_m2'], 'conditional R')
                full_miss = _spd(data['R_miss_proxy_m2'], 'miss proxy R')
                if full_R.shape != (len(ids), 2, 2) or full_miss.shape != full_R.shape:
                    raise ValueError('one covariance per artifact camera is required before masking')
                self.R = immutable_array(full_R[indices])
                self.R_miss = immutable_array(full_miss[indices])
                if np.linalg.eigvalsh(self.R_miss-self.R).min() < -1e-12:
                    raise ValueError('miss proxy cannot be more precise than conditional R')
                self.dynamic_R = schema == 'camera_network.thesis_stage09.v2'
            if self.dynamic_R:
                self.headings = np.asarray(data['headings'], dtype=float)
                if (self.headings.ndim != 1 or len(self.headings) < 3
                        or not np.isfinite(self.headings).all()
                        or not (np.diff(self.headings) > 0).all()
                        or abs(self.headings[0]) > 1e-12
                        or abs(self.headings[-1] - 2.0 * np.pi) > 1e-10):
                    raise ValueError('dynamic covariance headings must span [0,2pi]')
                full_field = _spd(data['R_cond_field_m2'], 'conditional R field')
                expected = (len(ids), len(self.headings), len(self.ys), len(self.xs), 2, 2)
                if full_field.shape != expected:
                    raise ValueError(f'conditional R field shape differs: {full_field.shape} != {expected}')
                self.R_field = immutable_array(full_field[indices])
        self.xs, self.ys = immutable_array(self.xs), immutable_array(self.ys)
        self.fields = MappingProxyType(self.fields)
        self.metadata = _freeze(self.metadata)
        if not self.direct_information:
            self.precision = immutable_array(np.linalg.solve(
                self.R, np.broadcast_to(np.eye(2), self.R.shape)))
            self.miss_precision = immutable_array(np.linalg.solve(
                self.R_miss, np.broadcast_to(np.eye(2), self.R.shape)))
        self.interpolators = {key: [RegularGridInterpolator((self.ys,self.xs), grid,
            bounds_error=False, fill_value=0.) for grid in maps] for key,maps in self.fields.items()}
        self.interpolators = MappingProxyType({key: tuple(value) for key,value in self.interpolators.items()})
        if self.direct_information:
            self.information_interpolators = tuple(
                tuple(tuple(
                    RegularGridInterpolator(
                        (self.ys, self.xs), field[..., row, column],
                        bounds_error=False, fill_value=0.0,
                    )
                    for column in range(2)) for row in range(2))
                for field in self.expected_information
            )
        else:
            self.information_interpolators = ()
        if self.dynamic_R:
            self.R_interpolators = tuple(
                tuple(tuple(
                    RegularGridInterpolator(
                        (self.headings, self.ys, self.xs), field[..., row, column],
                        bounds_error=False, fill_value=None,
                    )
                    for column in range(2)) for row in range(2))
                for field in self.R_field
            )
        else:
            self.R_interpolators = ()
        self._loaded = True

    @property
    def signature(self):
        return (self.metadata['schema'], self.sha256, self.camera_ids)

    def query(self, state):
        state = np.asarray(state, float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError('network query requires finite predicted [x,y,yaw]')
        out = {key:np.array([float(f([state[1],state[0]]).item()) for f in fs])
               for key,fs in self.interpolators.items()}
        if self.direct_information:
            matrices = []
            for camera in self.information_interpolators:
                matrix = np.asarray([
                    [float(camera[row][column]([state[1], state[0]]).item())
                     for column in range(2)] for row in range(2)
                ])
                matrices.append((matrix + matrix.T) / 2.0)
            out['expected_information'] = _psd(
                np.stack(matrices), 'interpolated expected information')
            return out
        if self.dynamic_R:
            yaw = float(state[2] % (2.0 * np.pi))
            matrices = []
            for camera in self.R_interpolators:
                matrix = np.asarray([
                    [float(camera[row][column]([yaw, state[1], state[0]]).item())
                     for column in range(2)] for row in range(2)
                ])
                matrices.append((matrix + matrix.T) / 2.0)
            out['conditional_covariance'] = _spd(
                np.stack(matrices), 'interpolated conditional R'
            )
        else:
            out['conditional_covariance'] = self.R
        return out

    def query_belief(self, state, P, kappa=1.):
        state, P = np.asarray(state,float), np.asarray(P,float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError('network query requires finite predicted [x,y,yaw]')
        P = validate_covariance(P)
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError('sigma-point kappa must be finite and positive')
        kappa = max(float(kappa), 1e-6)
        spread = np.sqrt(2+kappa)*np.linalg.cholesky((P[:2,:2]+P[:2,:2].T)/2+1e-9*np.eye(2))
        offsets = [np.zeros(2),spread[:,0],-spread[:,0],spread[:,1],-spread[:,1]]
        weights = np.asarray([kappa/(2+kappa)]+[1/(2*(2+kappa))]*4, dtype=float)
        # RegularGridInterpolator accepts a point batch.  Query only the score
        # and availability fields needed by the sigma-point average; calling
        # ``query`` here also interpolated every dynamic covariance element at
        # every sigma point even though those values were discarded.
        points_xy = state[None, :2] + np.asarray(offsets, dtype=float)
        points_yx = points_xy[:, ::-1]
        out = {
            key: np.asarray([
                float(np.dot(weights, interpolator(points_yx)))
                for interpolator in interpolators
            ], dtype=float)
            for key, interpolators in self.interpolators.items()
        }
        if self.direct_information:
            matrices = []
            for camera in self.information_interpolators:
                matrix = np.asarray([
                    [float(np.dot(weights, camera[row][column](points_yx)))
                     for column in range(2)] for row in range(2)
                ])
                matrices.append((matrix + matrix.T) / 2.0)
            out['expected_information'] = _psd(
                np.stack(matrices), 'belief-averaged expected information')
        return out

    def proxy_ground_covariance(self, score):
        """Designed IWAI precision blend; finite at a miss, never a runtime R."""
        if self.direct_information:
            raise RuntimeError('direct-information artifacts do not expose the legacy score proxy')
        score = np.asarray(score,float)
        if score.shape != (len(self.camera_ids),) or not np.isfinite(score).all() or np.any((score<0)|(score>1)):
            raise ValueError('one score in [0,1] per camera is required')
        info = (score[:,None,None]*self.precision + (1-score[:,None,None])*self.miss_precision).sum(axis=0)
        return np.linalg.solve(info,np.eye(2))

    def planning_diagnostics(self, state, P, H, kappa=1.):
        if self.direct_information:
            posterior, _ = self.expected_belief(state, P, kappa)
            queried = self.query_belief(state, P, kappa)
            support_key = (
                'residual_support'
                if 'residual_support' in queried else 'opportunity_support')
            support = queried[support_key]
            result = dict(
                p_vis=float('nan'), p_vis_eff=float('nan'),
                R_plan=np.asarray(posterior[:2, :2], dtype=float),
                r_plan_u_std=float(np.sqrt(posterior[0, 0])),
                r_plan_v_std=float(np.sqrt(posterior[1, 1])),
                network_artifact_sha256=self.sha256,
                p_vis_semantics='not_defined_for_camera_information_field',
            )
            result[
                'network_residual_support'
                if support_key == 'residual_support' else 'network_opportunity_support'
            ] = support
            return result
        query = self.query_belief(state,P,kappa)
        ground = self.proxy_ground_covariance(query['score'])
        J = projection_jacobian(H,state)
        uv = J @ ground @ J.T
        return dict(p_vis=float(query['score'].mean()), p_vis_eff=float(query['score'].mean()),
            R_plan=uv, r_plan_u_std=float(np.sqrt(uv[0,0])), r_plan_v_std=float(np.sqrt(uv[1,1])),
            network_score=query['score'], network_availability=query['availability'],
            network_R_proxy_m2=ground, network_artifact_sha256=self.sha256,
            p_vis_semantics='mean_expected_detector_score_not_detection_probability')

    def forecast_posterior(self, state, P, mode='branch'):
        """One-step availability reference, conditional independent camera errors.

        Uses q and R_cond, never the score or R_miss_proxy. Branch averaging is
        exact for this one-step linear model at the supplied predicted pose.
        Missingness is not treated as evidence about state. Temporal independence
        and the approximation to the actual robust runtime fusion are unvalidated.
        """
        query = self.query(state)
        if self.direct_information:
            if mode not in ('information', 'direct_information'):
                raise ValueError(
                    'direct-information artifacts support only the information approximation')
            P = validate_covariance(P, positive_definite=True, name='forecast prior')
            information = np.linalg.inv(P)
            information[:2, :2] += query['expected_information'].sum(axis=0)
            posterior = np.linalg.inv(information)
            return (posterior + posterior.T) / 2.0
        q = query['availability']
        conditional_R = query['conditional_covariance']
        P = validate_covariance(P, positive_definite=(mode == 'information'), name='forecast prior')
        if mode == 'information':
            info = np.linalg.solve(P,np.eye(3))
            info[:2,:2] += (q[:,None,None]*self.precision).sum(axis=0)
            return np.linalg.solve(info,np.eye(3))
        if mode != 'branch': raise ValueError('forecast mode must be branch or information')
        if len(q)>5: raise ValueError('exact reference is bounded to at most five cameras')
        out = np.zeros_like(P); H = np.eye(3)[:2]
        for mask in product((0,1),repeat=len(q)):
            weight = float(np.prod([q[i] if hit else 1-q[i] for i,hit in enumerate(mask)]))
            if weight == 0: continue
            post = P.copy()
            for hit,R in zip(mask,conditional_R):
                if not hit: continue
                K = np.linalg.solve(H @ post @ H.T+R,H @ post).T
                A = np.eye(3)-K @ H
                post = A @ post @ A.T+K @ R @ K.T
            out += weight*post
        return (out+out.T)/2

    def expected_belief(self, state, P, kappa=1., opportunities=1):
        """Expected posterior covariance and ambiguity after camera opportunities.

        Availability is averaged over the predicted XY belief. Camera hit events
        and conditional residuals are independent in this planning approximation.
        A miss performs no update. The robust runtime fusion remains a separate
        estimator and must not be inferred from this forecast.

        The returned ambiguity is ANCHORED: the expected posterior entropy minus
        the entropy of the posterior an ideally available step would reach from
        the same prior. Reporting the posterior entropy alone adds
        ``0.5 d log(2 pi e)`` plus a unit-dependent constant at every step, so
        under the arrival gate (which makes duration free) summing it charges
        ``constant * T`` -- a duration term whose sign is an artifact of units.

        The anchor is the ideal-availability posterior, NOT the prior.
        Differencing against the prior gives ``-q I(x;y)``, which is <= 0 and so
        pays a route for lasting longer, the same defect with the opposite sign.
        Against the ideal posterior the term is >= 0, dimensionless, and exactly
        zero for a perfectly observed step, so routes of different length are
        comparable -- which is what route selection needs.
        """
        if (isinstance(opportunities, bool) or int(opportunities) != opportunities
                or int(opportunities) < 1):
            raise ValueError('opportunities must be a positive integer')
        P = validate_covariance(P, positive_definite=True, name='network forecast prior')
        if self.direct_information:
            floor_information = np.zeros((3, 3), dtype=float)
            floor_information[:2, :2] = np.linalg.inv(
                AMBIGUITY_FLOOR_POSITION_SD_M ** 2 * np.eye(2))
            ambiguity = np.nan
            for _ in range(int(opportunities)):
                prior = P
                camera_information = self.query_belief(
                    state, prior, kappa)['expected_information']
                total = np.zeros((3, 3), dtype=float)
                total[:2, :2] = camera_information.sum(axis=0)
                P = np.linalg.inv(np.linalg.inv(prior) + total)
                P = (P + P.T) / 2.0
                ideal = np.linalg.inv(np.linalg.inv(prior) + floor_information)
                ideal = (ideal + ideal.T) / 2.0
                sign, logdet = np.linalg.slogdet(P[:2, :2])
                ideal_sign, ideal_logdet = np.linalg.slogdet(ideal[:2, :2])
                if sign <= 0 or ideal_sign <= 0:
                    raise ValueError('network posterior position covariance must be positive definite')
                ambiguity = float(max(0.5 * (logdet - ideal_logdet), 0.0))
            return P, ambiguity
        conditional_R = self.query(state)['conditional_covariance']
        if len(conditional_R) > 5:
            raise ValueError('exact expected-belief forecast is bounded to at most five cameras')
        # For each fixed hit subset, sequential linear-Gaussian updates equal
        # ``inv(inv(P) + sum(H.T @ inv(R_i) @ H))``.  Evaluate all 2^N subsets
        # as one NumPy batch.  This preserves exact branch enumeration while
        # avoiding millions of tiny Python-level solves during route selection.
        masks = _binary_report_masks(len(conditional_R))
        measurement_information = np.zeros((len(conditional_R), 3, 3), dtype=float)
        measurement_information[:, :2, :2] = np.linalg.inv(conditional_R)
        # One shared floor for every arm; see AMBIGUITY_FLOOR_POSITION_SD_M.
        floor_information = np.zeros((3, 3), dtype=float)
        floor_information[:2, :2] = np.linalg.inv(
            AMBIGUITY_FLOOR_POSITION_SD_M ** 2 * np.eye(2))
        expected_entropy = np.nan
        for _ in range(int(opportunities)):
            # Belief-averaged availability changes as the covariance contracts,
            # so each camera opportunity must query it from the current prior.
            P_prior = P
            q = np.clip(self.query_belief(state, P, kappa)['availability'], 0., 1.)
            weights = np.prod(
                np.where(masks != 0.0, q[None, :], 1.0 - q[None, :]), axis=1)
            branch_information = (
                np.linalg.inv(P)[None, :, :]
                + np.einsum('bc,cij->bij', masks, measurement_information)
            )
            posts = np.linalg.inv(branch_information)
            posts = 0.5 * (posts + posts.swapaxes(-1, -2))
            signs, logdets = np.linalg.slogdet(posts[:, :2, :2])
            if np.any(signs <= 0):
                raise ValueError('network posterior position covariance must be positive definite')
            # Anchor: the posterior an IDEALLY AVAILABLE step reaches from this
            # same prior (every camera reporting, each with its best
            # commissioned R). Anchoring to the PRIOR instead would give
            # -q*I(x;y) <= 0, which under the arrival gate pays a route for
            # lasting longer. Both entropies carry the same
            # 0.5*d*log(2*pi*e), so the difference is 0.5*log(|P|/|P_ideal|):
            # dimensionless, and zero for a perfectly observed step.
            # Same update formulation as the branch posteriors above
            # (information form), so each back-end is internally consistent and
            # the cross-back-end formulation gap cancels in the difference.
            ideal_post = np.linalg.inv(np.linalg.inv(P_prior) + floor_information)
            ideal_sign, ideal_logdet = np.linalg.slogdet(
                0.5 * (ideal_post + ideal_post.T)[:2, :2])
            if ideal_sign <= 0:
                raise ValueError('ideal-availability posterior must be positive definite')
            P = np.einsum('b,bij->ij', weights, posts)
            P = (P + P.T) / 2.
            expected_entropy = float(max(
                np.sum(weights * 0.5 * (2.0 * np.log(2.0 * np.pi * np.e) + logdets))
                - 0.5 * (2.0 * np.log(2.0 * np.pi * np.e) + ideal_logdet),
                0.0,
            ))
        return P, expected_entropy

    def effective_observation_covariance(self, state, P, kappa=1.):
        """Effective covariance obtained from the summed planning information.

        Current artifacts supply ``Lambda_i^plan`` directly. Historical
        artifacts reconstruct the same first-moment precision as
        ``q_i R_i^-1``. Only active cameras are present in either sum. This
        helper supports the ambiguity diagnostic; the belief rollout itself
        updates the prior covariance in information form.
        """
        if self.direct_information:
            precision = self.query_belief(
                state, P, kappa)['expected_information'].sum(axis=0)
        else:
            conditional_R = self.query(state)['conditional_covariance']
            availability = np.clip(
                self.query_belief(state, P, kappa)['availability'], 0., 1.)
            precision = np.zeros((2, 2), dtype=float)
            for weight, R in zip(availability, conditional_R):
                precision += float(weight) * np.linalg.inv(R)
        # Fixed ambiguity-only regularization. This keeps the inverse finite at
        # zero camera information without coupling the sensor term to Q.
        precision += AMBIGUITY_INFORMATION_REGULARIZER_M2_INV * np.eye(2)
        return np.linalg.inv(precision)

    def make_expected_belief_casadi(self, kappa=1., opportunities=1):
        """Return the differentiable counterpart of :meth:`expected_belief`."""
        import casadi as ca
        from planning.core.casadi_efe import (
            _differential_entropy_ca, _xy_visibility_sigma_points_ca,
        )
        if (isinstance(opportunities, bool) or int(opportunities) != opportunities
                or int(opportunities) < 1):
            raise ValueError('opportunities must be a positive integer')
        if self.direct_information:
            return self._make_direct_information_belief_casadi(
                ca, _differential_entropy_ca, _xy_visibility_sigma_points_ca,
                kappa, int(opportunities))
        if len(self.camera_ids) > 5:
            raise ValueError('exact expected-belief forecast is bounded to at most five cameras')
        interpolators = [
            ca.interpolant(
                f'network_availability_{self.sha256[:10]}_{i}', 'linear',
                [self.xs.tolist(), self.ys.tolist()], grid.T.ravel(order='F').tolist(),
            )
            for i, grid in enumerate(self.fields['availability'])
        ]
        # One shared floor for every arm; see AMBIGUITY_FLOOR_POSITION_SD_M.
        floor_R = ca.DM(AMBIGUITY_FLOOR_POSITION_SD_M ** 2 * np.eye(2))
        covariance_interpolators = None
        if self.dynamic_R:
            covariance_interpolators = [
                [[
                    ca.interpolant(
                        f'network_R_{self.sha256[:10]}_{camera}_{row}_{column}', 'linear',
                        [self.xs.tolist(), self.ys.tolist(), self.headings.tolist()],
                        np.transpose(self.R_field[camera, ..., row, column], (2, 1, 0))
                        .ravel(order='F').tolist(),
                    )
                    for column in range(2)] for row in range(2)]
                for camera in range(len(self.camera_ids))
            ]
        H = ca.DM(np.eye(3)[:2])

        def evaluate(m, P):
            conditional_R = []
            if covariance_interpolators is None:
                conditional_R = [ca.DM(value) for value in self.R]
            else:
                yaw = ca.atan2(ca.sin(m[2]), ca.cos(m[2]))
                yaw = ca.if_else(yaw < 0.0, yaw + 2.0 * np.pi, yaw)
                bounded = ca.vertcat(
                    ca.fmin(ca.fmax(m[0], self.xs[0]), self.xs[-1]),
                    ca.fmin(ca.fmax(m[1], self.ys[0]), self.ys[-1]),
                    yaw,
                )
                for camera in covariance_interpolators:
                    matrix = ca.vertcat(
                        ca.horzcat(camera[0][0](bounded), camera[0][1](bounded)),
                        ca.horzcat(camera[1][0](bounded), camera[1][1](bounded)),
                    )
                    conditional_R.append(.5 * (matrix + matrix.T))

            prior = .5 * (P + P.T)
            expected_entropy = 0.
            for _ in range(int(opportunities)):
                # Match the NumPy reference: re-average q over the contracted
                # belief before every opportunity within the global step.
                points, weights = _xy_visibility_sigma_points_ca(
                    m[:2], prior[:2, :2], kappa)
                availability = []
                for interp in interpolators:
                    total = 0.
                    for xy, weight in zip(points, weights):
                        inside = ca.logic_and(
                            ca.logic_and(xy[0] >= self.xs[0], xy[0] <= self.xs[-1]),
                            ca.logic_and(xy[1] >= self.ys[0], xy[1] <= self.ys[-1]),
                        )
                        bounded_xy = ca.vertcat(
                            ca.fmin(ca.fmax(xy[0], self.xs[0]), self.xs[-1]),
                            ca.fmin(ca.fmax(xy[1], self.ys[0]), self.ys[-1]),
                        )
                        total += float(weight) * ca.if_else(
                            inside, interp(bounded_xy), 0.)
                    availability.append(ca.fmin(ca.fmax(total, 0.), 1.))
                expected = ca.MX.zeros(3, 3)
                expected_entropy = 0.
                for mask in product((0, 1), repeat=len(availability)):
                    branch_weight = 1.
                    post = prior
                    for i, (hit, R_ca) in enumerate(zip(mask, conditional_R)):
                        q = availability[i]
                        branch_weight *= q if hit else 1. - q
                        if not hit:
                            continue
                        innovation = H @ post @ H.T + R_ca + 1e-9 * ca.DM.eye(2)
                        K = ca.solve(innovation, H @ post).T
                        A = ca.DM.eye(3) - K @ H
                        post = A @ post @ A.T + K @ R_ca @ K.T
                        post = .5 * (post + post.T)
                    expected += branch_weight * post
                    expected_entropy += branch_weight * _differential_entropy_ca(post[:2, :2])
                # Anchor: the posterior an IDEALLY AVAILABLE step would reach
                # from this same prior -- every camera reporting, each with its
                # best commissioned R. Anchoring to the PRIOR instead would give
                # -q*I(x;y), which is <= 0 and reintroduces the duration reward
                # with the opposite sign. See _anchored_ambiguity_ca.
                # Information form, matching the NumPy twin exactly. The
                # branch posteriors above use the Joseph form with a 1e-9
                # jitter; that jitter is negligible against a camera R but NOT
                # against this floor, which is three orders of magnitude
                # tighter, so a Joseph-form anchor disagrees with the NumPy
                # twin by ~4e-6 per opportunity and compounds. The anchor is a
                # fixed matrix, not a differentiated rollout quantity, so the
                # exact form costs nothing.
                floor_information = ca.MX.zeros(3, 3)
                floor_information[:2, :2] = ca.inv(floor_R)
                ideal = ca.inv(ca.inv(prior) + floor_information)
                ideal = .5 * (ideal + ideal.T)
                # Both terms are already 0.5*(D log 2*pi*e + logdet), so their
                # difference IS 0.5*log(|P| / |P_ideal|): the constant cancels.
                expected_entropy = ca.fmax(
                    expected_entropy - _differential_entropy_ca(ideal[:2, :2]), 0.0)
                prior = .5 * (expected + expected.T)
            return .5 * (expected + expected.T), expected_entropy

        return evaluate

    def make_effective_covariance_casadi(self, kappa=1.):
        """Differentiable counterpart of :meth:`effective_observation_covariance`."""
        import casadi as ca
        from planning.core.casadi_efe import _xy_visibility_sigma_points_ca
        if self.direct_information:
            return self._make_direct_effective_covariance_casadi(
                ca, _xy_visibility_sigma_points_ca, kappa)
        availability_interpolators = [
            ca.interpolant(
                f'network_reff_q_{self.sha256[:10]}_{i}', 'linear',
                [self.xs.tolist(), self.ys.tolist()], grid.T.ravel(order='F').tolist(),
            )
            for i, grid in enumerate(self.fields['availability'])
        ]
        covariance_interpolators = None
        if self.dynamic_R:
            covariance_interpolators = [
                [[
                    ca.interpolant(
                        f'network_reff_R_{self.sha256[:10]}_{camera}_{row}_{column}',
                        'linear',
                        [self.xs.tolist(), self.ys.tolist(), self.headings.tolist()],
                        np.transpose(self.R_field[camera, ..., row, column], (2, 1, 0))
                        .ravel(order='F').tolist(),
                    )
                    for column in range(2)] for row in range(2)]
                for camera in range(len(self.camera_ids))
            ]
        def evaluate(m, P):
            if covariance_interpolators is None:
                conditional_R = [ca.DM(value) for value in self.R]
            else:
                yaw = ca.atan2(ca.sin(m[2]), ca.cos(m[2]))
                yaw = ca.if_else(yaw < 0.0, yaw + 2.0 * np.pi, yaw)
                bounded = ca.vertcat(
                    ca.fmin(ca.fmax(m[0], self.xs[0]), self.xs[-1]),
                    ca.fmin(ca.fmax(m[1], self.ys[0]), self.ys[-1]),
                    yaw,
                )
                conditional_R = []
                for camera in covariance_interpolators:
                    matrix = ca.vertcat(
                        ca.horzcat(camera[0][0](bounded), camera[0][1](bounded)),
                        ca.horzcat(camera[1][0](bounded), camera[1][1](bounded)),
                    )
                    conditional_R.append(.5 * (matrix + matrix.T))
            points, weights = _xy_visibility_sigma_points_ca(m[:2], P[:2, :2], kappa)
            precision = (
                AMBIGUITY_INFORMATION_REGULARIZER_M2_INV * ca.DM.eye(2))
            for interp, R_ca in zip(availability_interpolators, conditional_R):
                total = 0.
                for xy, weight in zip(points, weights):
                    inside = ca.logic_and(
                        ca.logic_and(xy[0] >= self.xs[0], xy[0] <= self.xs[-1]),
                        ca.logic_and(xy[1] >= self.ys[0], xy[1] <= self.ys[-1]),
                    )
                    bounded_xy = ca.vertcat(
                        ca.fmin(ca.fmax(xy[0], self.xs[0]), self.xs[-1]),
                        ca.fmin(ca.fmax(xy[1], self.ys[0]), self.ys[-1]),
                    )
                    total += float(weight) * ca.if_else(inside, interp(bounded_xy), 0.)
                q = ca.fmin(ca.fmax(total, 0.), 1.)
                precision = precision + q * ca.inv(R_ca + 1e-12 * ca.DM.eye(2))
            return ca.inv(precision)

        return evaluate

    def _direct_information_interpolators_casadi(self, ca, prefix):
        return [
            [[
                ca.interpolant(
                    f'{prefix}_{self.sha256[:10]}_{camera}_{row}_{column}',
                    'linear', [self.xs.tolist(), self.ys.tolist()],
                    field[..., row, column].T.ravel(order='F').tolist(),
                )
                for column in range(2)] for row in range(2)]
            for camera, field in enumerate(self.expected_information)
        ]

    def _belief_averaged_information_casadi(
            self, ca, sigma_points, interpolators, m, P, kappa):
        points, weights = sigma_points(m[:2], P[:2, :2], kappa)
        total = ca.MX.zeros(2, 2)
        for camera in interpolators:
            matrix = ca.MX.zeros(2, 2)
            for row in range(2):
                for column in range(2):
                    value = 0.0
                    for xy, weight in zip(points, weights):
                        inside = ca.logic_and(
                            ca.logic_and(xy[0] >= self.xs[0], xy[0] <= self.xs[-1]),
                            ca.logic_and(xy[1] >= self.ys[0], xy[1] <= self.ys[-1]),
                        )
                        bounded = ca.vertcat(
                            ca.fmin(ca.fmax(xy[0], self.xs[0]), self.xs[-1]),
                            ca.fmin(ca.fmax(xy[1], self.ys[0]), self.ys[-1]),
                        )
                        value += float(weight) * ca.if_else(
                            inside, camera[row][column](bounded), 0.0)
                    matrix[row, column] = value
            total += 0.5 * (matrix + matrix.T)
        return total

    def _make_direct_information_belief_casadi(
            self, ca, differential_entropy, sigma_points, kappa, opportunities):
        interpolators = self._direct_information_interpolators_casadi(
            ca, 'network_direct_info')
        floor_information = ca.DM.zeros(3, 3)
        floor_information[:2, :2] = (
            1.0 / AMBIGUITY_FLOOR_POSITION_SD_M ** 2) * ca.DM.eye(2)

        def evaluate(m, P):
            prior = 0.5 * (P + P.T)
            ambiguity = 0.0
            for _ in range(opportunities):
                measurement = self._belief_averaged_information_casadi(
                    ca, sigma_points, interpolators, m, prior, kappa)
                state_information = ca.MX.zeros(3, 3)
                state_information[:2, :2] = measurement
                posterior = ca.inv(ca.inv(prior) + state_information)
                posterior = 0.5 * (posterior + posterior.T)
                ideal = ca.inv(ca.inv(prior) + floor_information)
                ideal = 0.5 * (ideal + ideal.T)
                ambiguity = ca.fmax(
                    differential_entropy(posterior[:2, :2])
                    - differential_entropy(ideal[:2, :2]),
                    0.0,
                )
                prior = posterior
            return prior, ambiguity

        return evaluate

    def _make_direct_effective_covariance_casadi(
            self, ca, sigma_points, kappa):
        interpolators = self._direct_information_interpolators_casadi(
            ca, 'network_direct_effective')
        def evaluate(m, P):
            precision = self._belief_averaged_information_casadi(
                ca, sigma_points, interpolators, m, P, kappa)
            precision += (
                AMBIGUITY_INFORMATION_REGULARIZER_M2_INV * ca.DM.eye(2))
            return ca.inv(precision)

        return evaluate

    def make_proxy_covariance_casadi(self, H, kappa=1.):
        if self.direct_information:
            raise RuntimeError(
                'direct-information artifacts do not expose the legacy score proxy')
        import casadi as ca
        from planning.core.casadi_efe import _xy_visibility_sigma_points_ca
        interpolators=[]
        for i,grid in enumerate(self.fields['score']):
            interpolators.append(ca.interpolant(f'network_{self.sha256[:10]}_{i}', 'linear',
                [self.xs.tolist(),self.ys.tolist()],grid.T.ravel(order='F').tolist()))
        H_numpy = np.asarray(H,float)
        if H_numpy.shape != (3,3) or not np.isfinite(H_numpy).all() or np.linalg.matrix_rank(H_numpy) != 3:
            raise ValueError('cost chart homography must be finite and nonsingular')
        H = ca.DM(H_numpy)
        def evaluate(m,P):
            points,weights = _xy_visibility_sigma_points_ca(m[:2],P[:2,:2],kappa)
            scores=[]
            for interp in interpolators:
                total=0
                for xy,weight in zip(points,weights):
                    inside=ca.logic_and(ca.logic_and(xy[0]>=self.xs[0],xy[0]<=self.xs[-1]),
                                        ca.logic_and(xy[1]>=self.ys[0],xy[1]<=self.ys[-1]))
                    bounded=ca.vertcat(ca.fmin(ca.fmax(xy[0],self.xs[0]),self.xs[-1]),
                                       ca.fmin(ca.fmax(xy[1],self.ys[0]),self.ys[-1]))
                    total += weight*ca.if_else(inside,interp(bounded),0.)
                scores.append(total)
            info=ca.DM.zeros(2,2)
            for score,visible,miss in zip(scores,self.precision,self.miss_precision):
                info += score*ca.DM(visible)+(1-score)*ca.DM(miss)
            ground=ca.solve(info,ca.DM.eye(2))
            projected=H @ ca.vertcat(m[0],m[1],1.)
            J=(H[:2,:2]*projected[2]-projected[:2] @ H[2,:2])/projected[2]**2
            # Same numerical domain as the NumPy chart. Invalid queries must
            # reach the finite-objective guard, not a fictitious finite chart.
            return ca.if_else(ca.fabs(projected[2]) >= 1e-8,
                              J @ ground @ J.T, ca.DM(np.full((2,2), np.nan)))
        return evaluate
