"""
Phase 1 — Offline Dataset Loader & Split

Loads a Minari offline RL dataset (e.g. mujoco/hopper/medium-v0) and produces
reproducible train / calibration / test / probe splits at the *transition*
level (s_t, a_t, s_{t+1}), matching the Master Research Plan's requirement
for a 1-step-ahead probabilistic world model.

Split roles (per the master plan):
    train       — used to fit the ensemble MLP world model
    calibration — held out, used to fit/verify calibration (e.g. temperature,
                  coverage adjustment) of predictive uncertainty
    test        — held out, used for final baseline metrics (NLL, coverage,
                  sharpness, calibration error) under the no-shift condition
    probe       — a fixed set of states used for policy intervention (P):
                  action is resampled from data_pi vs new_pi on the *same*
                  probe states, so it must remain disjoint and fixed across
                  runs (same seed -> same probe set every time)

All splits are done at the *episode* level first (so we never leak
transitions from the same trajectory across splits), then flattened to
transitions within each split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TransitionBatch:
    """A flat batch of (s_t, a_t, s_{t+1}) transitions."""

    observations: np.ndarray       # (N, obs_dim) -- s_t
    actions: np.ndarray            # (N, action_dim) -- a_t
    next_observations: np.ndarray  # (N, obs_dim) -- s_{t+1}
    episode_ids: np.ndarray        # (N,) -- which source episode each row came from

    def __len__(self) -> int:
        return self.observations.shape[0]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            next_observations=self.next_observations,
            episode_ids=self.episode_ids,
        )

    @classmethod
    def load(cls, path: Path) -> "TransitionBatch":
        data = np.load(path)
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            next_observations=data["next_observations"],
            episode_ids=data["episode_ids"],
        )


def _episodes_to_transitions(dataset, episode_ids: np.ndarray) -> TransitionBatch:
    """Flatten a set of Minari episodes into (s_t, a_t, s_{t+1}) transitions."""
    obs_list, act_list, next_obs_list, ep_id_list = [], [], [], []

    for ep in dataset.iterate_episodes(episode_indices=episode_ids.tolist()):
        obs = np.asarray(ep.observations)      # (T+1, obs_dim)
        act = np.asarray(ep.actions)           # (T, action_dim)
        T = act.shape[0]
        if T == 0:
            continue
        obs_list.append(obs[:T])
        act_list.append(act)
        next_obs_list.append(obs[1:T + 1])
        ep_id_list.append(np.full(T, ep.id, dtype=np.int64))

    return TransitionBatch(
        observations=np.concatenate(obs_list, axis=0).astype(np.float32),
        actions=np.concatenate(act_list, axis=0).astype(np.float32),
        next_observations=np.concatenate(next_obs_list, axis=0).astype(np.float32),
        episode_ids=np.concatenate(ep_id_list, axis=0),
    )


def load_and_split(
    dataset_id: str = "mujoco/hopper/medium-v0",
    seed: int = 1,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    test_frac: float = 0.10,
    probe_frac: float = 0.05,
    output_dir: str | Path = "data/processed",
) -> dict[str, TransitionBatch]:
    """
    Load a Minari dataset and produce reproducible episode-level splits.

    The split is done on EPISODE indices (not raw transitions) to avoid
    leaking adjacent timesteps of the same trajectory across splits, then
    each split is flattened into transitions.

    Returns a dict with keys: 'train', 'calibration', 'test', 'probe'.
    Also writes a manifest.json recording dataset id, seed, split sizes,
    and fractions, per the master plan's reproducibility requirement
    (Section 17: every final table/result must trace back to a
    machine-readable, reproducible artifact).
    """
    import minari

    assert abs(train_frac + calibration_frac + test_frac + probe_frac - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    dataset = minari.load_dataset(dataset_id, download=True)
    n_episodes = dataset.total_episodes

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)

    n_train = int(round(train_frac * n_episodes))
    n_cal = int(round(calibration_frac * n_episodes))
    n_test = int(round(test_frac * n_episodes))
    # probe gets the remainder to make sure all episodes are used

    idx_train = perm[:n_train]
    idx_cal = perm[n_train:n_train + n_cal]
    idx_test = perm[n_train + n_cal:n_train + n_cal + n_test]
    idx_probe = perm[n_train + n_cal + n_test:]

    splits = {
        "train": _episodes_to_transitions(dataset, idx_train),
        "calibration": _episodes_to_transitions(dataset, idx_cal),
        "test": _episodes_to_transitions(dataset, idx_test),
        "probe": _episodes_to_transitions(dataset, idx_probe),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_id": dataset_id,
        "seed": seed,
        "n_episodes_total": int(n_episodes),
        "n_episodes_per_split": {
            "train": int(len(idx_train)),
            "calibration": int(len(idx_cal)),
            "test": int(len(idx_test)),
            "probe": int(len(idx_probe)),
        },
        "n_transitions_per_split": {k: len(v) for k, v in splits.items()},
        "fractions": {
            "train": train_frac,
            "calibration": calibration_frac,
            "test": test_frac,
            "probe": probe_frac,
        },
        "observation_space": str(dataset.observation_space),
        "action_space": str(dataset.action_space),
    }

    for name, batch in splits.items():
        batch.save(output_dir / f"{name}.npz")

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return splits


if __name__ == "__main__":
    print("=== Phase 1: Loading and splitting dataset ===\n")
    splits = load_and_split(
        dataset_id="mujoco/hopper/medium-v0",
        seed=1,
        output_dir="data/processed/hopper_seed1",
    )
    print("Split sizes (transitions):")
    for name, batch in splits.items():
        print(f"  {name:<12} {len(batch):>8} transitions "
              f"({len(np.unique(batch.episode_ids))} episodes)")
    print("\nSaved to data/processed/hopper_seed1/")
    print("Manifest written to data/processed/hopper_seed1/manifest.json")
