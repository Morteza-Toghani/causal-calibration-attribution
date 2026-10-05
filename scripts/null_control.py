"""Null control: verify that identity interventions give ATE = 0.

For dynamics: run with mass_scale = 1.0. Re-simulated next states should
be identical (bit-for-bit) to the dataset's next states.
For policy: run with action_scale = 1.0. Same expectation.

If ATE != 0 for either, we have a re-simulation bias that would
contaminate the main analysis.

Run on all 5 seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import minari

from src.calibration.instance import instance_calibration_error
from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.interventions.dynamics import dynamics_next_state_from_env
from src.interventions.policy import action_scaler, resimulate_with_new_action
from src.model.ensemble_v2 import load_member, predict_ensemble


PAPER1 = Path.home() / 'paper1_run' / 'experiments' / 'hopper'


def run_null_seed(dataset, seed_idx, mechanism, scale):
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

    env = dataset.recover_environment()

    if mechanism == 'dynamics':
        n = min(len(test_obs), 5000)
        rng = np.random.default_rng(11000 + seed_idx)
        idx = np.sort(rng.choice(len(test_obs), n, replace=False))
        obs, act, y = test_obs[idx], test_act[idx], test_y[idx]

        pred = predict_ensemble(members, obs, act)
        base = instance_calibration_error(pred['mu'], pred['var'], y)

        y_new, _ = dynamics_next_state_from_env(
            env, obs, act, mass_scale=scale, seed=123 + seed_idx
        )
        pred_new = predict_ensemble(members, obs, act)
        treat = instance_calibration_error(pred_new['mu'], pred_new['var'], y_new)

        # Bit-exact check on re-simulation
        sim_diff = float(np.max(np.abs(y_new - y)))

    elif mechanism == 'policy':
        pred = predict_ensemble(members, probe_obs, probe_act)
        base = instance_calibration_error(pred['mu'], pred['var'], probe_y)

        action_low = env.action_space.low.astype(np.float32)
        action_high = env.action_space.high.astype(np.float32)
        transform = action_scaler(scale, action_low, action_high)
        new_act = transform(probe_act)
        y_new = resimulate_with_new_action(env, probe_obs, new_act)
        pred_new = predict_ensemble(members, probe_obs, new_act)
        treat = instance_calibration_error(pred_new['mu'], pred_new['var'], y_new)

        sim_diff = float(np.max(np.abs(y_new - probe_y)))

    env.close()

    diffs = treat - base
    ate = float(diffs.mean())
    return ate, sim_diff


def main():
    dataset = minari.load_dataset('mujoco/hopper/medium-v0', download=False)

    print(f'{"seed":<5} {"mechanism":<10} {"scale":<7} {"ATE":>15} {"max_|y_new - y|":>20}')
    print('-' * 65)
    all_ok = True
    for seed_idx in range(5):
        for mech, scale in (('dynamics', 1.0), ('policy', 1.0)):
            ate, sim_diff = run_null_seed(dataset, seed_idx, mech, scale)
            ate_ok = abs(ate) < 1e-8
            sim_ok = sim_diff < 1e-5
            all_ok &= (ate_ok and sim_ok)
            print(f'{seed_idx:<5} {mech:<10} {scale:<7.2f} {ate:>15.10f} {sim_diff:>20.2e}'
                  f'  {"OK" if ate_ok and sim_ok else "FAIL"}')

    print()
    print('ALL NULL CONTROLS PASS' if all_ok else 'SOME NULL CONTROLS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())