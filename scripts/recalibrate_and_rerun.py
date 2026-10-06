"""Post-hoc recalibration of baseline and re-run of all interventions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import minari

from src.calibration.instance import instance_calibration_error, z_for_central_coverage
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.interventions.dynamics import dynamics_next_state_from_env
from src.interventions.observation import compute_obs_scale, make_observation_noise
from src.interventions.policy import action_scaler, resimulate_with_new_action
from src.model.ensemble_v2 import load_member, predict_ensemble


PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'
NOMINAL = 0.90


def fit_alpha(mu, var, y, nominal=NOMINAL):
    z_nominal = z_for_central_coverage(nominal)

    def cov_at(alpha):
        sigma = np.sqrt(alpha * var)
        hw = z_nominal * sigma
        inside = (y >= mu - hw) & (y <= mu + hw)
        return float(inside.mean())

    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        c = cov_at(mid)
        if c < nominal:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def bootstrap_mean_ci(diffs, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(low), float(high)


def process_seed(dataset, seed_idx):
    legacy = PAPER1 / f'seed_{seed_idx:02d}'
    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000 + seed_idx,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )
    cal_obs, cal_act, cal_y = extract_transitions(
        dataset, split['calibration'], max_transitions=25000, seed=1000 + seed_idx + 1)
    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000, seed=1000 + seed_idx + 2)
    probe_obs, probe_act, probe_y = extract_transitions(
        dataset, split['probe'], max_transitions=10000, seed=1000 + seed_idx + 3)

    members = [load_member(legacy / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
               for i in range(5)]

    cal_pred = predict_ensemble(members, cal_obs, cal_act)
    alpha = fit_alpha(cal_pred['mu'], cal_pred['var'], cal_y, nominal=NOMINAL)
    print(f'  seed {seed_idx}: alpha = {alpha:.4f}')

    env = dataset.recover_environment()

    def pred_calibrated(obs, act, noise=None):
        p = predict_ensemble(members, obs, act, observation_noise=noise)
        p['var'] = alpha * p['var']
        return p

    rows = []

    base_pred = pred_calibrated(test_obs, test_act)
    base_inst = instance_calibration_error(base_pred['mu'], base_pred['var'], test_y)

    obs_scale = compute_obs_scale(test_obs)
    rng = np.random.default_rng(5000 + seed_idx)
    for level, nf in (('low', 0.05), ('high', 0.10)):
        eps = make_observation_noise(test_obs.shape, obs_scale, nf, rng)
        p = pred_calibrated(test_obs, test_act, noise=eps)
        treat = instance_calibration_error(p['mu'], p['var'], test_y)
        diffs = treat - base_inst
        ate, lo, hi = bootstrap_mean_ci(diffs, seed=7000 + seed_idx)
        rows.append({'seed': seed_idx, 'mechanism': 'observation', 'intensity': level,
                     'ate': ate, 'ci_low': lo, 'ci_high': hi, 'alpha': alpha})

    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    base_probe = pred_calibrated(probe_obs, probe_act)
    y_resim_1 = resimulate_with_new_action(env, probe_obs, probe_act)
    base_probe_inst = instance_calibration_error(
        base_probe['mu'], base_probe['var'], y_resim_1)

    for level, scale in (('low', 0.95), ('high', 0.80)):
        tf = action_scaler(scale, action_low, action_high)
        new_act = tf(probe_act)
        y_new = resimulate_with_new_action(env, probe_obs, new_act)
        p = pred_calibrated(probe_obs, new_act)
        treat = instance_calibration_error(p['mu'], p['var'], y_new)
        diffs = treat - base_probe_inst
        ate, lo, hi = bootstrap_mean_ci(diffs, seed=9000 + seed_idx)
        rows.append({'seed': seed_idx, 'mechanism': 'policy', 'intensity': level,
                     'ate': ate, 'ci_low': lo, 'ci_high': hi, 'alpha': alpha})

    dyn_n = min(len(test_obs), 5000)
    rng_dyn = np.random.default_rng(11000 + seed_idx)
    dyn_idx = np.sort(rng_dyn.choice(len(test_obs), dyn_n, replace=False))
    dyn_obs, dyn_act = test_obs[dyn_idx], test_act[dyn_idx]

    base_d = pred_calibrated(dyn_obs, dyn_act)
    y_dyn_1, _ = dynamics_next_state_from_env(
        env, dyn_obs, dyn_act, mass_scale=1.0, seed=123 + seed_idx)
    base_d_inst = instance_calibration_error(base_d['mu'], base_d['var'], y_dyn_1)

    for level, ms in (('low', 0.95), ('high', 0.80)):
        y_dyn, _ = dynamics_next_state_from_env(
            env, dyn_obs, dyn_act, mass_scale=ms, seed=123 + seed_idx)
        p = pred_calibrated(dyn_obs, dyn_act)
        treat = instance_calibration_error(p['mu'], p['var'], y_dyn)
        diffs = treat - base_d_inst
        ate, lo, hi = bootstrap_mean_ci(diffs, n_boot=2000, seed=12000 + seed_idx)
        rows.append({'seed': seed_idx, 'mechanism': 'dynamics', 'intensity': level,
                     'ate': ate, 'ci_low': lo, 'ci_high': hi, 'alpha': alpha})

    env.close()
    return rows


def main():
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)
    all_rows = []
    for seed_idx in range(5):
        print(f'processing seed {seed_idx}...')
        all_rows.extend(process_seed(dataset, seed_idx))

    df = pd.DataFrame(all_rows)
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'recalibrated_interventions.csv', index=False)
    print(f'saved {out_dir / "recalibrated_interventions.csv"}')

    agg = df.groupby(['mechanism', 'intensity']).agg(
        ate_mean=('ate', 'mean'),
        ate_sd=('ate', 'std'),
        alpha_mean=('alpha', 'mean'),
    ).reset_index()

    print()
    print('=== Recalibrated ATE (mean over 5 seeds) ===')
    print(f'{"mechanism":<14} {"intensity":<8} {"ate":>12} {"sd":>10} {"alpha":>10}')
    print('-' * 60)
    for _, r in agg.iterrows():
        print(f"{r['mechanism']:<14} {r['intensity']:<8} "
              f"{r['ate_mean']:>+12.5f} {r['ate_sd']:>10.5f} {r['alpha_mean']:>10.4f}")


if __name__ == '__main__':
    main()