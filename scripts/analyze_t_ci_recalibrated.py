"""t-based CI for recalibrated ATEs."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


def t_ci(values, alpha=0.05):
    x = np.asarray(values, dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    t_crit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))
    return m, m - t_crit * se, m + t_crit * se


def main():
    df = pd.read_csv('results/tables/recalibrated_interventions.csv')
    rows = []
    for (mech, level), g in df.groupby(['mechanism', 'intensity']):
        vals = g['ate'].values
        m, lo, hi = t_ci(vals)
        rows.append({'mechanism': mech, 'intensity': level,
                     'mean': m, 'lo': lo, 'hi': hi,
                     'excl0': (lo > 0) or (hi < 0)})
    out = pd.DataFrame(rows).sort_values(['mechanism', 'intensity']).reset_index(drop=True)
    out.to_csv('results/tables/recalibrated_t_ci.csv', index=False)
    print('saved results/tables/recalibrated_t_ci.csv')
    print()
    print('=== t-based 95% CI on recalibrated ATE (n=5, t_crit=2.776) ===')
    print(f'{"condition":<22} {"mean":>10} {"t-CI 95%":>28} {"excl0":>8}')
    print('-' * 72)
    for _, r in out.iterrows():
        ci = f'[{r["lo"]:+.5f}, {r["hi"]:+.5f}]'
        print(f'{r["mechanism"]+"_"+r["intensity"]:<22} {r["mean"]:>+10.5f} {ci:>28} {("YES" if r["excl0"] else "NO"):>8}')


if __name__ == '__main__':
    main()