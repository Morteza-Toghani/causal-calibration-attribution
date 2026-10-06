"""Alpha sensitivity analysis for Paper 1 recalibrated ATE.

Sweeps the variance-scaling factor alpha over a fixed grid and recomputes
the per-condition ATE as a function of alpha. Directly addresses the
reviewer concern that the observation-dominance result is an artifact
of the particular alpha fitted on the calibration split.

The fitted alpha (from the main recalibrated pipeline) is marked on the
plot as a reference line.

Outputs:
  results/tables/alpha_sensitivity_per_seed.csv
  results/tables/alpha_sensitivity_summary.csv
  results/figures/alpha_sensitivity.png
  results/figures/alpha_sensitivity.pdf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import minari
from scipy import stats as _stats

from src.calibration.frozen_metrics import z_for_central_coverage
from src.calibration.instance import instance_calibration_error
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.interventions.dynamics import dynamics_next_state_from_env
from src.interventions.observation import compute_obs_scale, make_observation_noise
from src.interventions.policy import action_scaler, resimulate_with_new_action
from src.model.ensemble_v2 import load_member, predict_ensemble


DATASET_IDS = {
    'hopper': 'mujoco/hopper/medium-v0',
    'walker2d': 'mujoco/walker2d/medium-v0',
}
LEGACY_HOPPER = Path.home() / 'paper1_run' / 'experiments' / 'hopper'
NEW_ROOT = Path('results/ensemble_training').resolve()
NOMINAL_RECAL = 0.90

ALPHA_GRID = np.round(np.arange(0.05, 1.05, 0.05), 4)
FITTED_ALPHA_MEAN = {'hopper': 0.2361, 'walker2d': 0.3743}


def ckpt_root(env_name, seed_idx):
    if env_name == 'hopper' and seed_idx in range(5):
        return LEGACY_HOPPER / f'seed_{seed_idx:02d}' / 'checkpoints'
    return NEW_ROOT / env_name / f'seed_{seed_idx:02d}' / 'checkpoints'


def fit_alpha(mu, var, y, nominal=NOMINAL_RECAL):
    z_nominal = z_for_central_coverage(nominal)
    def cov_at(alpha):
        sigma = np.sqrt(alpha * var)
        hw = z_nominal * sigma
        return float(((y >= mu - hw) & (y <= mu + hw)).mean())
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if cov_at(mid) < nominal:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def t_ci_95(values):
    n = len(values)
    m = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else float('nan')
    se = sd / np.sqrt(n) if n > 1 else float('nan')
    t_crit = float(_stats.t.ppf(0.975, df=n - 1)) if n > 1 else float('nan')
    return m, m - t_crit * se, m + t_crit * se


def process_seed(env_name, dataset, seed_idx, alpha_grid):
    root = ckpt_root(env_name, seed_idx)
    members = [load_member(root / f'ensemble_member_{i:02d}.pt') for i in range(5)]

    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000 + seed_idx,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )
    cal_obs, cal_act, cal_y = extract_transitions(
        dataset, split['calibration'], max_transitions=25000,
        seed=1000 + seed_idx + 1)
    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000,
        seed=1000 + seed_idx + 2)
    probe_obs, probe_act, probe_y = extract_transitions(
        dataset, split['probe'], max_transitions=10000,
        seed=1000 + seed_idx + 3)

    # Fitted alpha (reference)
    cal_pred = predict_ensemble(members, cal_obs, cal_act)
    alpha_fit = fit_alpha(cal_pred['mu'], cal_pred['var'], cal_y)

    env = dataset.recover_environment()
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    # --- Baseline predictions (unscaled variance) ---
    pred_o = predict_ensemble(members, test_obs, test_act)
    base_o_mu, base_o_var, base_o_y = pred_o['mu'], pred_o['var'], test_y

    dyn_n = min(len(test_obs), 5000)
    rng_dyn = np.random.default_rng(11000 + seed_idx)
    dyn_idx = np.sort(rng_dyn.choice(len(test_obs), dyn_n, replace=False))
    dyn_obs, dyn_act = test_obs[dyn_idx], test_act[dyn_idx]
    pred_d_base = predict_ensemble(members, dyn_obs, dyn_act)
    y_dyn_resim_1, _ = dynamics_next_state_from_env(
        env, dyn_obs, dyn_act, mass_scale=1.0, seed=123 + seed_idx)
    base_d_mu, base_d_var, base_d_y = pred_d_base['mu'], pred_d_base['var'], y_dyn_resim_1

    pred_p_base = predict_ensemble(members, probe_obs, probe_act)
    y_probe_resim_1 = resimulate_with_new_action(env, probe_obs, probe_act)
    base_p_mu, base_p_var, base_p_y = pred_p_base['mu'], pred_p_base['var'], y_probe_resim_1

    # --- Treatments ---
    # Observation: only noise changes; y = test_y (unchanged)
    obs_scale = compute_obs_scale(test_obs)
    rng_obs = np.random.default_rng(5000 + seed_idx)
    obs_treat = {}
    for level, nf in (('low', 0.05), ('high', 0.10)):
        eps = make_observation_noise(test_obs.shape, obs_scale, nf, rng_obs)
        p = predict_ensemble(members, test_obs, test_act, observation_noise=eps)
        obs_treat[level] = (p['mu'], p['var'], test_y)

    # Dynamics: same (mu, var) as baseline, only y changes
    dyn_treat = {}
    for level, ms in (('low', 0.95), ('high', 0.80)):
        y_dyn, _ = dynamics_next_state_from_env(
            env, dyn_obs, dyn_act, mass_scale=ms, seed=123 + seed_idx)
        dyn_treat[level] = (base_d_mu, base_d_var, y_dyn)

    # Policy: actions change -> (mu, var, y) all differ from baseline
    policy_treat = {}
    for level, scale in (('low', 0.95), ('high', 0.80)):
        tf = action_scaler(scale, action_low, action_high)
        new_act = tf(probe_act)
        y_new = resimulate_with_new_action(env, probe_obs, new_act)
        p = predict_ensemble(members, probe_obs, new_act)
        policy_treat[level] = (p['mu'], p['var'], y_new)

    env.close()

    # --- Sweep alpha ---
    rows = []
    for alpha in alpha_grid:
        base_o_ce = instance_calibration_error(base_o_mu, alpha * base_o_var, base_o_y)
        for level in ('low', 'high'):
            mu_t, var_t, y_t = obs_treat[level]
            tr = instance_calibration_error(mu_t, alpha * var_t, y_t)
            rows.append({
                'env': env_name, 'seed': seed_idx,
                'mechanism': 'observation', 'intensity': level,
                'alpha': float(alpha), 'alpha_fit': alpha_fit,
                'ate': float((tr - base_o_ce).mean()),
            })

        base_d_ce = instance_calibration_error(base_d_mu, alpha * base_d_var, base_d_y)
        for level in ('low', 'high'):
            mu_t, var_t, y_t = dyn_treat[level]
            tr = instance_calibration_error(mu_t, alpha * var_t, y_t)
            rows.append({
                'env': env_name, 'seed': seed_idx,
                'mechanism': 'dynamics', 'intensity': level,
                'alpha': float(alpha), 'alpha_fit': alpha_fit,
                'ate': float((tr - base_d_ce).mean()),
            })

        base_p_ce = instance_calibration_error(base_p_mu, alpha * base_p_var, base_p_y)
        for level in ('low', 'high'):
            mu_t, var_t, y_t = policy_treat[level]
            tr = instance_calibration_error(mu_t, alpha * var_t, y_t)
            rows.append({
                'env': env_name, 'seed': seed_idx,
                'mechanism': 'policy', 'intensity': level,
                'alpha': float(alpha), 'alpha_fit': alpha_fit,
                'ate': float((tr - base_p_ce).mean()),
            })

    return rows


def summarize(per_seed_df):
    out = []
    keys = ['env', 'mechanism', 'intensity', 'alpha']
    for key, grp in per_seed_df.groupby(keys):
        env, mech, lvl, alpha = key
        diffs = grp['ate'].values
        mean, lo, hi = t_ci_95(diffs)
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float('nan')
        out.append({
            'env': env, 'mechanism': mech, 'intensity': lvl, 'alpha': alpha,
            'n': len(diffs), 'mean': mean, 'sd': sd,
            'ci_low': lo, 'ci_high': hi,
        })
    return pd.DataFrame(out)


def make_plot(summary, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    styles = {
        ('observation', 'low'):  ('#1f77b4', '-',  'O low (5%)'),
        ('observation', 'high'): ('#1f77b4', '--', 'O high (10%)'),
        ('dynamics', 'low'):     ('#d62728', '-',  'D low (0.95)'),
        ('dynamics', 'high'):    ('#d62728', '--', 'D high (0.80)'),
        ('policy', 'low'):       ('#2ca02c', '-',  'P low (0.95)'),
        ('policy', 'high'):      ('#2ca02c', '--', 'P high (0.80)'),
    }
    for ax, env_name in zip(axes, ['hopper', 'walker2d']):
        sub = summary[summary['env'] == env_name]
        for (mech, lvl), (color, ls, label) in styles.items():
            s = sub[(sub['mechanism'] == mech) &
                    (sub['intensity'] == lvl)].sort_values('alpha')
            if len(s) == 0:
                continue
            ax.plot(s['alpha'], s['mean'], color=color, linestyle=ls,
                    marker='o', markersize=3, label=label)
            ax.fill_between(s['alpha'], s['ci_low'], s['ci_high'],
                            color=color, alpha=0.10)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(FITTED_ALPHA_MEAN[env_name], color='gray',
                   linestyle=':', linewidth=1.2,
                   label=f'fitted α = {FITTED_ALPHA_MEAN[env_name]:.3f}')
        ax.set_xlabel(r'variance scaling factor $\alpha$')
        ax.set_title(env_name)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('ATE (instance CE, treat - base)')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle('Sensitivity of ATE to variance-scaling factor $\\alpha$')
    fig.tight_layout()
    fig.savefig(fig_dir / 'alpha_sensitivity.png', dpi=150, bbox_inches='tight')
    fig.savefig(fig_dir / 'alpha_sensitivity.pdf', bbox_inches='tight')
    plt.close(fig)


def main(envs):
    t0 = time.time()
    out_dir = Path('results/tables')
    fig_dir = Path('results/figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for env_name in envs:
        print(f"\n=== {env_name} ===")
        dataset = minari.load_dataset(DATASET_IDS[env_name])
        for seed_idx in range(10):
            t_s = time.time()
            rows = process_seed(env_name, dataset, seed_idx, ALPHA_GRID)
            all_rows.extend(rows)
            print(f"  seed {seed_idx}... {time.time() - t_s:.1f}s")

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(out_dir / 'alpha_sensitivity_per_seed.csv', index=False)
    print(f"\nsaved: {out_dir / 'alpha_sensitivity_per_seed.csv'}")

    summary = summarize(per_seed)
    summary.to_csv(out_dir / 'alpha_sensitivity_summary.csv', index=False)
    print(f"saved: {out_dir / 'alpha_sensitivity_summary.csv'}")

    # --- Console preview: ATE at fitted alpha vs at alpha=1.0 ---
    print("\n--- ATE at fitted alpha vs alpha = 1.0 ---")
    for env_name in envs:
        sub = summary[summary['env'] == env_name]
        for mech in ['observation', 'dynamics', 'policy']:
            for lvl in ['low', 'high']:
                s = sub[(sub['mechanism'] == mech) &
                        (sub['intensity'] == lvl)].sort_values('alpha')
                if len(s) == 0:
                    continue
                a_fit = FITTED_ALPHA_MEAN[env_name]
                idx_fit = (s['alpha'] - a_fit).abs().idxmin()
                idx_one = (s['alpha'] - 1.0).abs().idxmin()
                print(f"  {env_name:9s} {mech:12s} {lvl:4s}  "
                      f"alpha={s.loc[idx_fit,'alpha']:.2f}: "
                      f"{s.loc[idx_fit,'mean']:+.5f}  |  "
                      f"alpha=1.00: {s.loc[idx_one,'mean']:+.5f}")

    try:
        make_plot(summary, fig_dir)
        print(f"\nsaved: {fig_dir / 'alpha_sensitivity.png'}")
        print(f"saved: {fig_dir / 'alpha_sensitivity.pdf'}")
    except Exception as e:
        print(f"plot failed: {e}")

    print(f"\ntotal time: {(time.time() - t0)/60:.1f} min")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=['hopper', 'walker2d', 'both'],
                   default='both')
    args = p.parse_args()
    envs = ['hopper', 'walker2d'] if args.env == 'both' else [args.env]
    main(envs)
