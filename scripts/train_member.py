"""Train one ensemble member (Hopper or Walker2d) with FROZEN hyperparams.

Usage:
    python scripts/train_member.py --env walker2d --seed_idx 0 --member_idx 0

Resume: if the checkpoint file already exists, exits early (unless --force).
Device: CPU-only, threads set via --threads (default 2 for 5-way parallel).
Output format matches paper1_run/.../ensemble_member_XX.pt exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Set threads BEFORE importing torch
_parser = argparse.ArgumentParser()
_parser.add_argument('--env', required=True, choices=['hopper', 'walker2d'])
_parser.add_argument('--seed_idx', type=int, required=True)
_parser.add_argument('--member_idx', type=int, required=True)
_parser.add_argument('--threads', type=int, default=2)
_parser.add_argument('--out_root', default='results/ensemble_training')
_parser.add_argument('--force', action='store_true')
args = _parser.parse_args()

os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ['MKL_NUM_THREADS'] = str(args.threads)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import minari

from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.model.standardizer import fit_standardizer, apply_standardizer


DATASET_IDS = {
    'hopper': 'mujoco/hopper/medium-v0',
    'walker2d': 'mujoco/walker2d/medium-v0',
}


def build_model_class():
    class ProbabilisticMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden=(256, 256)):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            self.backbone = nn.Sequential(*layers)
            self.mu_head = nn.Linear(prev, output_dim)
            self.logvar_head = nn.Linear(prev, output_dim)

        def forward(self, x):
            h = self.backbone(x)
            mu = self.mu_head(h)
            raw_logvar = self.logvar_head(h)
            logvar = torch.clamp(raw_logvar, min=-10.0, max=5.0)
            return mu, logvar

    return ProbabilisticMLP


def gaussian_nll(mu, logvar, target):
    var = torch.exp(logvar)
    return 0.5 * (logvar + (target - mu) ** 2 / var + math.log(2.0 * math.pi)).mean()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    torch.set_num_threads(args.threads)

    seed_idx = args.seed_idx
    member_idx = args.member_idx
    env_name = args.env
    member_seed = seed_idx * 100 + member_idx
    split_seed = 1000 + seed_idx

    seed_root = Path(args.out_root) / env_name / f'seed_{seed_idx:02d}'
    ckpt_dir = seed_root / 'checkpoints'
    log_dir = seed_root / 'logs'
    ckpt_path = ckpt_dir / f'ensemble_member_{member_idx:02d}.pt'

    if ckpt_path.exists() and not args.force:
        print(f'[skip] {env_name} seed_{seed_idx:02d} member_{member_idx:02d}: '
              f'{ckpt_path} exists')
        return

    print(f'[start] {env_name} seed={seed_idx} member={member_idx} '
          f'member_seed={member_seed} threads={args.threads}')
    t0 = time.time()

    dataset = minari.load_dataset(DATASET_IDS[env_name], download=False)

    # Deterministic split
    split = deterministic_episode_split(
        dataset.total_episodes, seed=split_seed,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )

    # Save split manifest (only if first member of this seed)
    split_manifest_path = seed_root / 'split_manifest.json'
    if not split_manifest_path.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(split_manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                'env': env_name,
                'dataset_id': DATASET_IDS[env_name],
                'seed_idx': seed_idx,
                'split_seed': split_seed,
                'train_episode_ids': split['train'],
                'calibration_episode_ids': split['calibration'],
                'test_episode_ids': split['test'],
                'probe_episode_ids': split['probe'],
            }, f)

    # Extract train transitions (FROZEN cap 150k, seed = split_seed)
    train_obs, train_act, train_y = extract_transitions(
        dataset, split['train'], max_transitions=150_000, seed=split_seed)

    # Standardizer on TRAIN
    std = fit_standardizer(train_obs, train_act, train_y)
    xz, yz = apply_standardizer(std, train_obs, train_act, train_y)

    X = torch.from_numpy(xz)
    Y = torch.from_numpy(yz)

    g = torch.Generator()
    g.manual_seed(member_seed)
    loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=1024, shuffle=True, generator=g,
        num_workers=0, pin_memory=False,
    )

    set_global_seed(member_seed)
    Model = build_model_class()
    obs_dim = train_obs.shape[1]
    act_dim = train_act.shape[1]
    model = Model(obs_dim + act_dim, obs_dim, hidden=(256, 256))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    history = []
    for epoch in range(1, 26):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            mu, logvar = model(xb)
            loss = gaussian_nll(mu, logvar, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        history.append({'epoch': epoch, 'train_nll': mean_loss})
        if epoch == 1 or epoch % 5 == 0:
            print(f'  epoch {epoch:>2}/25 train_nll={mean_loss:.4f}')

    elapsed = time.time() - t0

    # Save checkpoint in FROZEN format
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'standardizer': asdict(std),
        'input_dim': X.shape[1],
        'output_dim': Y.shape[1],
        'hidden': [256, 256],
        'member_seed': member_seed,
        'tag': f'ensemble_member_{member_idx:02d}',
    }, ckpt_path)

    # Save log
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f'member_{member_idx:02d}.json', 'w', encoding='utf-8') as f:
        json.dump({
            'env': env_name,
            'seed_idx': seed_idx,
            'member_idx': member_idx,
            'member_seed': member_seed,
            'split_seed': split_seed,
            'device': 'cpu',
            'torch_version': torch.__version__,
            'cpu_count': os.cpu_count(),
            'threads': args.threads,
            'elapsed_seconds': elapsed,
            'n_train_transitions': int(len(train_obs)),
            'obs_dim': int(obs_dim),
            'action_dim': int(act_dim),
            'history': history,
        }, f, indent=2)

    # Update training_manifest (append member)
    manifest_path = seed_root / 'training_manifest.json'
    if manifest_path.exists():
        manifest = json.load(open(manifest_path, encoding='utf-8'))
    else:
        manifest = {
            'env': env_name,
            'dataset_id': DATASET_IDS[env_name],
            'seed_idx': seed_idx,
            'split_seed': split_seed,
            'device': 'cpu',
            'torch_version': torch.__version__,
            'hidden': [256, 256],
            'epochs': 25,
            'batch_size': 1024,
            'learning_rate': 3e-4,
            'weight_decay': 1e-5,
            'grad_clip_max_norm': 5.0,
            'logvar_clamp': [-10.0, 5.0],
            'optimizer': 'AdamW',
            'members': [],
        }
    manifest['members'].append({
        'member_idx': member_idx,
        'member_seed': member_seed,
        'elapsed_seconds': elapsed,
        'checkpoint': str(ckpt_path),
        'final_train_nll': history[-1]['train_nll'],
    })
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'[done] {ckpt_path} in {elapsed:.1f}s  '
          f'final_nll={history[-1]["train_nll"]:.4f}')


if __name__ == '__main__':
    main()