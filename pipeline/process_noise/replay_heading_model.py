"""Offline EKF replay of logged campaign runs: camera_xy_only vs coupled heading.

Open loop: the robot path, odometry and fused camera measurements are the logged ones.
Only the estimator differs. Prediction uses the planner's own unicycle_step,
unicycle_jacobian and unicycle_process_noise (planning.core.dynamics) with the run's
config (dt, process_noise_xy, process_noise_theta). Update: position measurement H = [I 0],
fused z and R from fusion_observations.csv, NIS gate 9.21, Joseph form.

xy_only reproduces the node: after an accepted update the heading mean is the map-frame
odometry heading, heading/position cross terms are zeroed and P_thth keeps its predicted
value. At read-out the node re-anchors heading variance to q_theta^2 * t_since_odom_start.
coupled: plain EKF, heading corrected through its correlation with position.
xy_consistent: xy_only, but the internal heading variance is the same odometry-heading
variance the node reports at read-out (q^2 * t since odometry start, floor 0.5 deg), set at
the start and after every accepted update, so prediction and read-out use one heading model.
coupled_every: heading is a filter state, never reset to odometry. An accepted batch at
least COUPLED_PERIOD_S after the last heading-correcting one is a full EKF update; every
other accepted batch updates position only (gain row for heading zeroed; the Joseph form
keeps the covariance exact for that gain). Starts from the odometry heading variance.
coupled_confirm: like coupled_every with period 0, but a fix corrects heading only if it
agrees with the previous accepted fix: (z_k - z_prev) minus the odometry displacement of the
mean between them is inside the 95% ellipse of R_k + R_prev. Otherwise position only.
coupled_lookahead: a fix corrects heading only if at least LOOKAHEAD_MIN camera batches
follow within the next LOOKAHEAD_S (offline look-ahead = the best a fixed-lag version can do).

Timing follows the node: a batch is applied at its image (capture) stamp, from the anchor,
but becomes visible only at its apply_stamp (correction_assimilations.csv); a read-out
predicts from the newest visible anchor.

Known answer: xy_only must reproduce the logged planner belief (est_*).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src/planning'))
from planning.core.encoder_noise_model import ENCODER_PSD  # noqa: E402
PSD = None
from planning.core.dynamics import unicycle_jacobian, unicycle_process_noise, unicycle_step  # noqa: E402

GATE = 9.21
OUT = REPO / 'logs/thesis/analysis/process_noise'
COUPLED_PERIOD_S = 1.0
LOOKAHEAD_S = 1.0
LOOKAHEAD_MIN = 4
Q_THETA_OVERRIDE = None   # set from the command line: --q-theta
GATE_LOG = []              # one row per camera batch the replay considers
SUB_DT = 0.05
H = np.hstack([np.eye(2), np.zeros((2, 1))])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_run(exp: pathlib.Path):
    cfg = yaml.safe_load(next(exp.glob('campaign_seed*.yaml')).read_text())
    s = json.loads((exp / 'run_summary.json').read_text())
    e = pd.read_csv(exp / 'experiment.csv')
    odo = (e[['odom_noisy_stamp', 'odom_noisy_v', 'odom_noisy_w', 'odom_noisy_yaw']].dropna()
           .drop_duplicates('odom_noisy_stamp').sort_values('odom_noisy_stamp'))
    out = e[['planner_belief_stamp', 'est_x', 'est_y', 'est_yaw', 'est_cov_xx', 'est_cov_xy',
             'est_cov_yy', 'planner_cov_yaw']].dropna()
    out = out[out.planner_belief_stamp >= float(s['first_cmd_stamp'])]
    out = out.drop_duplicates('planner_belief_stamp').sort_values('planner_belief_stamp')
    f = pd.read_csv(exp / 'fusion_observations.csv')
    f = f[f.n_used >= 1].drop_duplicates('source_batch_id').sort_values('common_capture_stamp')
    c = pd.read_csv(exp / 'correction_assimilations.csv')
    first = c[c.accepted == 1].sort_values('correction_stamp').iloc[0]
    g = pd.read_csv(exp / 'ground_truth_pose.csv').sort_values('stamp_s')
    return cfg, s, odo, out, f, first, g


def replay(exp: pathlib.Path, mode: str) -> pd.DataFrame:
    cfg, s, odo, out, fus, first, g = load_run(exp)
    qxy, qth, base_dt = float(cfg['process_noise_xy']), float(cfg['process_noise_theta']), float(cfg['dt'])
    if Q_THETA_OVERRIDE is not None:
        qth = Q_THETA_OVERRIDE
    ot, ov, ow = odo.odom_noisy_stamp.to_numpy(), odo.odom_noisy_v.to_numpy(), odo.odom_noisy_w.to_numpy()
    # map-frame odometry heading: odometry yaw plus the commissioned spawn heading offset
    g_yaw0 = float(np.interp(ot[0], g.stamp_s, np.unwrap(g.yaw)))
    oyaw = np.unwrap(odo.odom_noisy_yaw.to_numpy()) + (g_yaw0 - odo.odom_noisy_yaw.iloc[0])
    odom_origin = ot[0]

    def odom_heading(t):
        return wrap(float(np.interp(t, ot, oyaw)))

    def predict(m, P, t_from, t_to):
        t = t_from
        while t < t_to - 1e-9:
            i = max(np.searchsorted(ot, t, side='right') - 1, 0)
            nxt = ot[i + 1] if i + 1 < len(ot) else np.inf
            dt = min(t_to, nxt, t + SUB_DT) - t
            u = np.array([ov[i], ow[i]])
            F = unicycle_jacobian(m, u, dt)
            Q = unicycle_process_noise(qxy, qth, dt, theta=float(m[2]), v=float(u[0]), base_dt=base_dt, w=float(u[1]), psd=PSD)
            m = np.asarray(unicycle_step(m, u, dt), dtype=float)
            P = F @ P @ F.T + Q
            t += dt
        return m, 0.5 * (P + P.T)

    t_anchor = float(first.correction_stamp)
    m = np.array(json.loads(first.posterior_mean), dtype=float)
    P = np.array(json.loads(first.posterior_covariance), dtype=float)

    if PSD is not None:   # accumulated sigma_w^2(v, w) along the odometry, as the node does
        from planning.core.encoder_noise_model import encoder_psd
        rate = encoder_psd(ov, ow, PSD)[1]
        cumvar = np.concatenate([[0.0], np.cumsum(rate[1:] * np.diff(ot))])
    def odom_heading_var(t):
        v = (float(np.interp(t, ot, cumvar)) if PSD is not None else qth ** 2 * (t - odom_origin))
        return min(np.pi ** 2, max(np.radians(0.5) ** 2, v))

    if mode in ('xy_consistent', 'coupled_every', 'coupled_confirm', 'coupled_lookahead'):
        P[:2, 2] = P[2, :2] = 0.0
        P[2, 2] = odom_heading_var(t_anchor)
    last_coupled = -np.inf
    prev_fix = None   # (z, R, mean xy right after that update)
    apply = pd.read_csv(exp / 'correction_assimilations.csv')[['source_batch_id', 'apply_stamp']]
    meas = fus.merge(apply, on='source_batch_id', how='inner')
    meas = meas[meas.common_capture_stamp > t_anchor]
    queries = out.planner_belief_stamp.to_numpy()
    events = sorted([(q, 1, None) for q in queries if q > t_anchor]
                    + [(r.apply_stamp, 0, r) for r in meas.itertuples()],
                    key=lambda e: (e[0], e[1]))
    rows, last_acc, n_rej = [], t_anchor, 0
    caps = np.sort(meas.common_capture_stamp.to_numpy())
    last_acc_c = t_anchor
    for te, kind, r in events:
        if kind == 0:
            c = float(r.common_capture_stamp)
            if c < t_anchor:          # older than the anchor: the node refuses it too
                continue
            m, P = predict(m, P, t_anchor, c)
            t_anchor = c
            z = np.array([r.fused_x, r.fused_y])
            R = np.array([[r.fused_cov_xx, r.fused_cov_xy], [r.fused_cov_xy, r.fused_cov_yy]])
            Sy = H @ P @ H.T + R
            nu = z - H @ m
            nis = float(nu @ np.linalg.solve(Sy, nu))
            gx_c, gy_c = np.interp(c, g.stamp_s, g.x), np.interp(c, g.stamp_s, g.y)
            GATE_LOG.append(dict(run=str(exp), mode=mode, batch=r.source_batch_id, capture=c,
                                 t_run=c - float(s['first_cmd_stamp']), gap_before=c - last_acc_c,
                                 nis=nis, accepted=nis <= GATE, n_cams=int(r.n_used),
                                 meas_err_m=float(np.hypot(z[0] - gx_c, z[1] - gy_c)),
                                 pred_err_m=float(np.hypot(m[0] - gx_c, m[1] - gy_c)),
                                 pred_sig_major_m=float(np.sqrt(np.linalg.eigvalsh(P[:2, :2])[-1])),
                                 R_sig_major_m=float(np.sqrt(np.linalg.eigvalsh(R)[-1])),
                                 gx=gx_c, gy=gy_c))
            if nis <= GATE:
                last_acc_c = c
                K = P @ H.T @ np.linalg.inv(Sy)
                if mode == 'coupled_lookahead':
                    ahead = np.sum((caps > c) & (caps <= c + LOOKAHEAD_S))
                    if ahead < LOOKAHEAD_MIN:
                        K[2, :] = 0.0
                if mode == 'coupled_confirm':
                    ok = False
                    if prev_fix is not None:
                        dz = (z - prev_fix[0]) - (m[:2] - prev_fix[2])
                        ok = float(dz @ np.linalg.solve(R + prev_fix[1], dz)) <= 5.991
                    if not ok:
                        K[2, :] = 0.0
                if mode == 'coupled_every':
                    if c - last_coupled >= COUPLED_PERIOD_S:
                        last_coupled = c
                    else:
                        K[2, :] = 0.0        # position-only update between heading corrections
                P_pred_thth = P[2, 2]
                m = m + K @ nu
                IKH = np.eye(3) - K @ H
                P = IKH @ P @ IKH.T + K @ R @ K.T
                if mode in ('xy_only', 'xy_consistent'):
                    m[2] = odom_heading(c)
                    P[:2, 2] = P[2, :2] = 0.0
                    P[2, 2] = P_pred_thth if mode == 'xy_only' else odom_heading_var(c)
                m[2] = wrap(m[2])
                prev_fix = (z, R, m[:2].copy())
                last_acc = te
            else:
                n_rej += 1
        else:
            mo, Po = predict(m.copy(), P.copy(), t_anchor, te)
            if mode in ('xy_only', 'xy_consistent'):    # read-out anchor, _anchor_belief_yaw_for_planning
                mo[2] = odom_heading(te)
                Po[:2, 2] = Po[2, :2] = 0.0
                Po[2, 2] = odom_heading_var(te)
            rows.append((te, te - last_acc, n_rej, *mo, Po[0, 0], Po[0, 1], Po[1, 1], Po[2, 2]))
    df = pd.DataFrame(rows, columns=['t', 'gap', 'n_rej', 'x', 'y', 'yaw', 'cxx', 'cxy', 'cyy', 'cthth'])
    df['t_run'] = df.t - float(s['first_cmd_stamp'])
    df['gx'] = np.interp(df.t, g.stamp_s, g.x)
    df['gy'] = np.interp(df.t, g.stamp_s, g.y)
    df['gyaw'] = np.interp(df.t, g.stamp_s, np.unwrap(g.yaw))
    return df.merge(out.rename(columns={'planner_belief_stamp': 't'}), on='t', how='left')


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = np.stack([df.x - df.gx, df.y - df.gy], -1)
    P = np.stack([np.stack([df.cxx, df.cxy], -1), np.stack([df.cxy, df.cyy], -1)], -2)
    df = df.copy()
    df['nees'] = np.einsum('ni,nij,nj->n', d, np.linalg.inv(P), d)
    n = np.stack([-np.sin(df.gyaw), np.cos(df.gyaw)], -1)
    df['ct_err'] = np.einsum('ni,ni->n', d, n)
    df['ct_sig'] = np.sqrt(np.einsum('ni,nij,nj->n', n, P, n))
    df['h_err'] = np.degrees(wrap(df.yaw - df.gyaw))
    df['h_sig'] = np.degrees(np.sqrt(df.cthth))
    return df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--q-theta', type=float, default=None, help='override process_noise_theta')
    ap.add_argument('--modes', nargs='+', default=['xy_only', 'coupled'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--encoder-q', action='store_true')
    ap.add_argument('--campaign-log', type=pathlib.Path, default=None,
                    help='read runs from a campaign_log.json instead of logs/thesis/analysis/runs.csv')
    ap.add_argument('--coupled-period', type=float, default=1.0)
    ap.add_argument('only', nargs='*', help='optional run_dir substrings')
    args = ap.parse_args()
    Q_THETA_OVERRIDE = args.q_theta
    PSD = ENCODER_PSD if args.encoder_q else None
    COUPLED_PERIOD_S = args.coupled_period
    if args.campaign_log is not None:
        import json as _json
        rows = []
        for entry in _json.loads(args.campaign_log.read_text()).values():
            if entry.get('outcome') in (None, 'infra_invalid') or not entry.get('run_log_dir'):
                continue
            exp = sorted(pathlib.Path(entry['run_log_dir']).glob('experiment_*'))
            if not exp or not (exp[0] / 'run_summary.json').exists():
                continue
            rows.append(dict(task=entry['task'], condition=entry['condition'], seed=entry['seed'],
                             model=entry['condition'].rsplit('_', 1)[0], state=entry['condition'].rsplit('_', 1)[1],
                             run_dir=str(exp[0])))
        runs = pd.DataFrame(rows)
    else:
        runs = pd.read_csv(REPO / 'logs/thesis/analysis/runs.csv')
    parts = []
    for _, r in runs.iterrows():
        if not isinstance(r.run_dir, str):
            continue
        if args.only and not any(o in r.run_dir for o in args.only):
            continue
        for mode in args.modes:
            df = metrics(replay(REPO / r.run_dir, mode))
            df['mode'], df['cond'], df['state'], df['task'], df['seed'] = mode, r.condition, r.state, r.task, r.seed
            parts.append(df)
    all_df = pd.concat(parts)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'replay_heading_mode{args.tag}.parquet'
    all_df.to_parquet(out)
    pd.DataFrame(GATE_LOG).to_parquet(out.with_name(f'gate_log{args.tag}.parquet'))
    print('wrote', out, len(all_df))
