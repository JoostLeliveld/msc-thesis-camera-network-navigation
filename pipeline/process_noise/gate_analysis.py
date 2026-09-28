"""Where do the extra NIS-gate rejections of xy_consistent + q 0.036 happen, and do they hurt?"""
import glob, pathlib
import numpy as np, pandas as pd
REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / 'logs/thesis/analysis/process_noise'
pd.set_option('display.width', 220)
a = pd.read_parquet(OUT / 'gate_log_q008.parquet'); b = pd.read_parquet(OUT / 'gate_log_cons_q0358.parquet')
# known answer: replay baseline rejections vs the node's own logged rejections
logged = 0
for p in glob.glob(str(REPO / 'logs/thesis/campaign') + '/seed*/*/*/seed*/attempts/*/experiment_*/correction_assimilations.csv'):
    c = pd.read_csv(p); logged += int((c.status == 'rejected').sum())
print(f'rejections: node logged {logged} | replay now {int((~a.accepted).sum())} | replay fix+q0.036 {int((~b.accepted).sum())}')
m = a.merge(b, on=['run', 'batch'], suffixes=('_now', '_new'))
m['task'] = m.run.str.extract(r'(thesis10_[a-z_]+?)/(?:global|per_camera|spatial)')[0].str.replace('thesis10_', '')
m['cond'] = m.run.str.extract(r'/((?:global|per_camera|spatial)_(?:intact|removal))/')[0]
extra = m[m.accepted_now & ~m.accepted_new]
fewer = m[~m.accepted_now & m.accepted_new]
print(f'batches compared {len(m)}; rejected only in new {len(extra)}; rejected only now {len(fewer)}')
print('\nEXTRA REJECTIONS: by task/condition'); print(extra.groupby(['task', 'cond']).size().to_string())
print('\nEXTRA REJECTIONS: what they were (medians)')
cols = ['t_run_new', 'gap_before_new', 'nis_new', 'nis_now', 'n_cams_new', 'meas_err_m_new', 'pred_err_m_new', 'pred_sig_major_m_new', 'R_sig_major_m_new']
print(extra[cols].describe(percentiles=[.5]).loc[['count', 'mean', '50%', 'max']].round(3).to_string())
print('\nshare where the rejected measurement was CLOSER to truth than the prediction (harmful):',
      round(np.mean(extra.meas_err_m_new < extra.pred_err_m_new), 3))
print('share where the measurement error > 3x its own R sigma (a bad measurement, rejection right):',
      round(np.mean(extra.meas_err_m_new > 3 * extra.R_sig_major_m_new), 3))
print('\ncompare: all accepted batches in new, median meas_err', round(m.loc[m.accepted_new, 'meas_err_m_new'].median(), 3),
      ' median pred_err', round(m.loc[m.accepted_new, 'pred_err_m_new'].median(), 3))
# consequence: longest stretch without an accepted update, per run
def longest_gap(df, col):
    out = {}
    for run, g in df.groupby('run'):
        t = np.sort(g.loc[g[col], 'capture'].to_numpy())
        out[run] = np.diff(t).max() if len(t) > 1 else np.nan
    return pd.Series(out)
lg = pd.DataFrame(dict(now=longest_gap(a, 'accepted'), new=longest_gap(b, 'accepted')))
d = lg.new - lg.now
print(f'\nlongest stretch without accepted update per run: runs longer in new {int((d > 0.05).sum())}, '
      f'median extra {d[d > 0.05].median():.2f} s, max extra {d.max():.2f} s')
print(lg.assign(extra=d).sort_values('extra', ascending=False).head(6).round(2).to_string())
print('\nlocations of extra rejections (x, y rounded to 1 m) most frequent:')
print(extra.assign(x=extra.gx_new.round(0), y=extra.gy_new.round(0)).groupby(['x', 'y']).size().sort_values(ascending=False).head(8).to_string())
