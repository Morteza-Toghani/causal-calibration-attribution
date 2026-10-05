"""Dynamics intervention — FROZEN pipeline semantics.

Matches Paper1 pipeline lines 765-834 (dynamics_next_state_from_env).

A single physical parameter (MuJoCo body_mass) is scaled during
evaluation. The observation process and action-selection are kept
fixed.

The simulator state is reconstructed from each observation:
  - Hopper-v5  : qpos[1:] = o[:5],  qvel = o[5:11]
  - Walker2d-v5: qpos[1:] = o[:8],  qvel = o[8:17]
  - x position is set to 0 (not present in the observation)
"""
from __future__ import annotations

import numpy as np


SPEC_CONFIG = {
    'Hopper-v5':   {'nq': 6, 'nv': 6, 'qpos_tail_len': 5, 'qvel_slice': (5, 11)},
    'Walker2d-v5': {'nq': 9, 'nv': 9, 'qpos_tail_len': 8, 'qvel_slice': (8, 17)},
}


def _reconstruct_state(obs_row: np.ndarray, spec_id: str):
    cfg = SPEC_CONFIG[spec_id]
    qpos = np.zeros(cfg['nq'], dtype=np.float64)
    qvel = np.zeros(cfg['nv'], dtype=np.float64)
    n = cfg['qpos_tail_len']
    qpos[1:] = obs_row[:n]
    qvel[:] = obs_row[cfg['qvel_slice'][0]:cfg['qvel_slice'][1]]
    return qpos, qvel


def dynamics_next_state_from_env(
    env,
    observations: np.ndarray,
    actions: np.ndarray,
    mass_scale: float,
    seed: int = 0,
):
    """Counterfactual next states under scaled body mass.

    Returns:
        next_obs: np.ndarray  shape (N, obs_dim)
        audit:    list of dicts with 'terminated', 'truncated'
    """
    spec_id = env.spec.id
    if spec_id not in SPEC_CONFIG:
        raise ValueError(f'Unsupported environment for dynamics intervention: {spec_id}')

    model = env.unwrapped.model
    if not hasattr(model, 'body_mass'):
        raise RuntimeError('MuJoCo model does not expose body_mass.')

    base_mass = model.body_mass.copy()
    model.body_mass[:] = base_mass * mass_scale

    results = []
    try:
        for o, a in zip(observations, actions):
            qpos, qvel = _reconstruct_state(o, spec_id)
            env.unwrapped.set_state(qpos, qvel)
            next_obs, _, terminated, truncated, _ = env.unwrapped.step(
                a.astype(np.float32)
            )
            results.append({
                'next_obs': np.asarray(next_obs, dtype=np.float32),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
            })
    finally:
        model.body_mass[:] = base_mass
        env.reset(seed=seed)

    return np.stack([r['next_obs'] for r in results], axis=0), results
