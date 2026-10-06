"""t-based CI for n=5 seeds."""
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
    df = pd.read_csv('results/tables/interventions_corrected.csv')
    rows = []
    for (mech, level), g in df.groupby(['mechanism', 'intensity']):
        vals = g['ate_corrected'].values
        vals_orig = g['ate_original'].values
        m_c, lo_c, hi_c = t_ci(vals)
        m_o, lo_o, hi_o = t_ci(vals_orig)
        rows.append({
            'mechanism': mech, 'intensity': level,
            'mean_corrected': m_c, 'ci_low_corrected': lo_c, 'ci_high_corrected': hi_c,
            'excl0_corr': (lo_c > 0) or (hi_c < 0),
            'mean_original': m_o, 'ci_low_original': lo_o, 'ci_high_original': hi_o,
            'excl0_orig': (lo_o > 0) or (hi_o < 0),
        })
    out = pd.DataFrame(rows).sort_values(['mechanism', 'intensity']).reset_index(drop=True)
    out.to_csv('results/tables/interventions_corrected_t_ci.csv', index=False)
    print('saved results/tables/interventions_corrected_t_ci.csv')

    md = ['# t-based 95% CI on ATE (n=5, t_crit=2.776)', '']
    md.append('| condition | ATE (corrected) | t-CI 95% | excl 0? |')
    md.append('|---|---|---|---|')
    for _, r in out.iterrows():
        ci = f'[{r["ci_low_corrected"]:+.5f}, {r["ci_high_corrected"]:+.5f}]'
        mark = 'YES' if r['excl0_corr'] else 'NO'
        md.append(f'| {r["mechanism"]}_{r["intensity"]} | {r["mean_corrected"]:+.5f} | {ci} | {mark} |')
    md.append('')
    md.append('## Original ATE for comparison')
    md.append('')
    md.append('| condition | ATE (orig) | t-CI 95% | excl 0? |')
    md.append('|---|---|---|---|')
    for _, r in out.iterrows():
        ci = f'[{r["ci_low_original"]:+.5f}, {r["ci_high_original"]:+.5f}]'
        mark = 'YES' if r['excl0_orig'] else 'NO'
        md.append(f'| {r["mechanism"]}_{r["intensity"]} | {r["mean_original"]:+.5f} | {ci} | {mark} |')
    with open('results/tables/interventions_t_ci_comparison.md', 'w', encoding='utf-8') as f:
        f.write(chr(10).join(md))
    print('saved results/tables/interventions_t_ci_comparison.md')

    print()
    print('=== t-CI corrected ===')
    print(f'{"condition":<22} {"mean":>10} {"t-CI":>28} {"excl 0?":>10}')
    print('-' * 74)
    for _, r in out.iterrows():
        ci = f'[{r["ci_low_corrected"]:+.5f}, {r["ci_high_corrected"]:+.5f}]'
        mark = 'YES' if r['excl0_corr'] else 'NO'
        print(f'{r["mechanism"] + "_" + r["intensity"]:<22} {r["mean_corrected"]:>+10.5f} {ci:>28} {mark:>10}')


if __name__ == '__main__':
    main()
