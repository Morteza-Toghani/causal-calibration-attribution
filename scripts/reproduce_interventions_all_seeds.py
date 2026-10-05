"""Reproduce intervention_records.json for all 5 Hopper seeds."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import minari

from src.calibration.instance import instance_calibration_error
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.interventions.dynamics import dynamics_next_state_from_env
from src.interventions.observation import compute_obs_scale, make_observation_noise
from src.interventions.policy import action_scaler, resimulate_with_new_action
from src.model.ensemble_v2 import load_member, predict_ensemble


PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'


def bootstrap_mean_ci(diffs, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(low), float(high)


def paired_permutation_pvalue(diffs, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    observed = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    perm_means = np.abs((signs * d).mean(axis=1))
    return float((1.0 + np.sum(perm_means >= observed)) / (n_perm + 1.0))


def reproduce_seed(dataset, seed_idx):
    legacy = PAPER1 / f'seed_{seed_idx:02d}'

    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000 + seed_idx,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )

    test_obs, test_act, test_y = extract_transitions(
        dataset, split['test'], max_transitions=25000, seed=1000 + seed_idx + 2
    )
    probe_obs, probe_act, probe_y = extract_transitions(
        dataset, split['probe'], max_transitions=10000, seed=1000 + seed_idx + 3
    )

    members = [load_member(legacy / 'checkpoints' / f'ensemble_member_{i:02d}.pt')
               for i in range(5)]

    base_pred = predict_ensemble(members, test_obs, test_act)
    base_instance = instance_calibration_error(base_pred['mu'], base_pred['var'], test_y)

    records = []

    # Observation
    obs_scale = compute_obs_scale(test_obs)
    rng = np.random.default_rng(5000 + seed_idx)
    for level, noise_fraction in (('low', 0.05), ('high', 0.10)):
        eps = make_observation_noise(test_obs.shape, obs_scale, noise_fraction, rng)
        pred_o = predict_ensemble(members, test_obs, test_act, observation_noise=eps)
        treat = instance_calibration_error(pred_o['mu'], pred_o['var'], test_y)
        diffs = treat - base_instance
        ate, lo, hi = bootstrap_mean_ci(diffs, n_boot=3000, seed=7000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=5000, seed=8000 + seed_idx)
        records.append({
            'mechanism': 'observation', 'intensity': level,
            'ate': ate, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
        })

    # Policy
    env = dataset.recover_environment()
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    base_probe_pred = predict_ensemble(members, probe_obs, probe_act)
    base_probe_instance = instance_calibration_error(
        base_probe_pred['mu'], base_probe_pred['var'], probe_y
    )

    for level, scale in (('low', 0.95), ('high', 0.80)):
        transform = action_scaler(scale, action_low, action_high)
        new_act = transform(probe_act)
        new_y = resimulate_with_new_action(env, probe_obs, new_act)
        pred_p = predict_ensemble(members, probe_obs, new_act)
        treat = instance_calibration_error(pred_p['mu'], pred_p['var'], new_y)
        diffs = treat - base_probe_instance
        ate, lo, hi = bootstrap_mean_ci(diffs, n_boot=3000, seed=9000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=5000, seed=10000 + seed_idx)
        records.append({
            'mechanism': 'policy', 'intensity': level,
            'ate': ate, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
        })

    # Dynamics
    dyn_n = min(len(test_obs), 5000)
    rng_dyn = np.random.default_rng(11000 + seed_idx)
    dyn_idx = np.sort(rng_dyn.choice(len(test_obs), dyn_n, replace=False))
    dyn_obs = test_obs[dyn_idx]
    dyn_act = test_act[dyn_idx]
    dyn_base_y = test_y[dyn_idx]

    base_dyn_pred = predict_ensemble(members, dyn_obs, dyn_act)
    base_dyn_instance = instance_calibration_error(
        base_dyn_pred['mu'], base_dyn_pred['var'], dyn_base_y
    )

    for level, mass_scale in (('low', 0.95), ('high', 0.80)):
        dyn_y, audit = dynamics_next_state_from_env(
            env, dyn_obs, dyn_act, mass_scale=mass_scale, seed=123 + seed_idx
        )
        pred_d = predict_ensemble(members, dyn_obs, dyn_act)
        treat = instance_calibration_error(pred_d['mu'], pred_d['var'], dyn_y)
        diffs = treat - base_dyn_instance
        ate, lo, hi = bootstrap_mean_ci(diffs, n_boot=2000, seed=12000 + seed_idx)
        p = paired_permutation_pvalue(diffs, n_perm=3000, seed=13000 + seed_idx)
        records.append({
            'mechanism': 'dynamics', 'intensity': level,
            'ate': ate, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
        })

    env.close()

    with open(legacy / 'results' / 'intervention_records.json', encoding='utf-8') as f:
        target = json.load(f)

    return records, target


def main():
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)

    print(f'{"seed":<5} {"condition":<20} {"ate_reproduced":>17} {"ate_target":>17} {"rel_err":>10}')
    print('-' * 75)

    all_ok = True
    for seed_idx in range(5):
        records, target = reproduce_seed(dataset, seed_idx)
        for r, t in zip(records, target):
            cond = f"{r['mechanism']}_{r['intensity']}"
            rel = abs(r['ate'] - t['ate']) / (abs(t['ate']) + 1e-12)
            ok = rel < 1e-6
            all_ok &= ok
            marker = '✓' if ok else '✗'
            print(f'{seed_idx:<5} {cond:<20} {r["ate"]:>17.10f} {t["ate"]:>17.10f} '
                  f'{rel:>10.2e} {marker}')
        print()

    print('ALL 5 SEEDS PASS' if all_ok else 'SOME SEEDS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
