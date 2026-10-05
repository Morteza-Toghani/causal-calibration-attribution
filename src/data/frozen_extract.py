"""FROZEN pipeline transition extraction — exact copy from Paper1 pipeline."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def extract_episode_arrays(ep):
    obs = np.asarray(ep.observations)
    actions = np.asarray(ep.actions)
    rewards = np.asarray(ep.rewards)
    terminations = np.asarray(ep.terminations)
    truncations = np.asarray(ep.truncations)

    if obs.shape[0] != actions.shape[0] + 1:
        raise ValueError(
            f'Unexpected episode alignment: observations={obs.shape[0]}, '
            f'actions={actions.shape[0]}. Expected obs=T+1 and actions=T.'
        )
    return obs, actions, rewards, terminations, truncations


def extract_transitions(
    dataset,
    episode_indices: Iterable[int],
    max_transitions: int | None = None,
    seed: int = 42,
):
    X_obs, X_act, Y_next = [], [], []

    for ep_idx in episode_indices:
        ep = dataset[int(ep_idx)]
        obs, actions, _, _, _ = extract_episode_arrays(ep)
        X_obs.append(obs[:-1])
        X_act.append(actions)
        Y_next.append(obs[1:])

    if not X_obs:
        raise ValueError('No episodes supplied for transition extraction.')

    obs = np.concatenate(X_obs, axis=0).astype(np.float32)
    act = np.concatenate(X_act, axis=0).astype(np.float32)
    nxt = np.concatenate(Y_next, axis=0).astype(np.float32)

    if max_transitions is not None and len(obs) > max_transitions:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(obs), size=max_transitions, replace=False)
        take.sort()
        obs = obs[take]
        act = act[take]
        nxt = nxt[take]

    return obs, act, nxt
