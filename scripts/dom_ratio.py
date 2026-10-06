import pandas as pd
df = pd.read_csv('results/tables/alpha_sensitivity_summary.csv')

def get(env, mech, lvl, alpha):
    sub = df[(df.env == env) & (df.mechanism == mech) & (df.intensity == lvl)]
    return float(sub.iloc[(sub.alpha - alpha).abs().argmin()]['mean'])

rows = []
for env in ['hopper', 'walker2d']:
    for alpha in [0.10, 0.25, 0.35, 0.50, 0.75, 1.00]:
        a = round(alpha, 2)
        ol, dl = get(env, 'observation', 'low', a),  get(env, 'dynamics', 'low', a)
        oh, dh = get(env, 'observation', 'high', a), get(env, 'dynamics', 'high', a)
        rows.append({
            'env': env, 'alpha': a,
            'O_low': ol, 'D_low': dl, 'r_low': abs(ol)/max(abs(dl), 1e-9),
            'O_high': oh, 'D_high': dh, 'r_high': abs(oh)/max(abs(dh), 1e-9),
        })
out = pd.DataFrame(rows)
out.to_csv('results/tables/dom_ratio_by_alpha.csv', index=False)
print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
