import pandas as pd, numpy as np
import pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / 'logs/thesis/analysis/process_noise'
pd.set_option('display.width', 220)
d = pd.read_parquet(OUT / 'replay_heading_mode.parquet')
x = d[d['mode'] == 'xy_only']
r = np.sqrt(x.cxx) / np.sqrt(x.est_cov_xx)
print(f'KNOWN ANSWER all runs (xy_only vs logged): sigma_x ratio median {r.median():.3f} '
      f'p5 {r.quantile(.05):.3f} p95 {r.quantile(.95):.3f}; pos diff median '
      f'{100*np.hypot(x.x-x.est_x, x.y-x.est_y).median():.2f} cm; pooled in95 {np.mean(x.nees<=5.991):.3f}')
d['gapbin'] = pd.cut(d.gap, [-1, 0.5, 1, 2, 4, 8, 100], labels=['<0.5', '0.5-1', '1-2', '2-4', '4-8', '>8'])
def summ(g):
    return pd.Series(dict(n=len(g), in95=np.mean(g.nees <= 5.991),
        ct_sig_cm=100 * g.ct_sig.median(), ct_err_cm=100 * g.ct_err.abs().median(),
        ct_z_rms=np.sqrt(np.mean((g.ct_err / g.ct_sig) ** 2)),
        h_sig=g.h_sig.median(), h_err=g.h_err.abs().median(), h_err_p95=g.h_err.abs().quantile(.95),
        h_z_rms=np.sqrt(np.mean((g.h_err / g.h_sig) ** 2))))
print('\nPOOLED'); print(d.groupby('mode').apply(summ, include_groups=False).round(3).to_string())
print('\nBY GAP'); print(d.groupby(['mode', 'gapbin'], observed=True).apply(summ, include_groups=False).round(3).to_string())
tb = pd.cut(d.t_run, [0, 5, 10, 20, 30, 60])
print('\nHEADING BY RUN TIME'); print(d.groupby(['mode', tb], observed=True).apply(summ, include_groups=False)[['n','h_sig','h_err','h_err_p95','h_z_rms']].round(2).to_string())
print('\nWORST HEADING ERROR PER RUN (coupled vs xy_only), deg')
w = d.groupby(['mode','task','cond','seed']).h_err.apply(lambda s: s.abs().max()).unstack(0)
print(w.describe().round(2).to_string())
