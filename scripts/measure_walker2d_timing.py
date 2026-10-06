"""Measure timing for a single Walker2d ensemble member.

Runs 2 epochs with the exact FROZEN hyperparameters on Walker2d and
extrapolates. This is a TIMING MEASUREMENT only — it does not save
checkpoints or affect the repository.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import minari

from src.data.frozen_extract import extract_transitions
from src.data.frozen_split import deterministic_episode_split
from src.model.standardizer import fit_standardizer, apply_standardizer


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
    print('=' * 70)
    print('WALKER2D TIMING MEASUREMENT (2 epochs -> extrapolate to 25)')
    print('=' * 70)

    cpu = os.cpu_count()
    print(f'os.cpu_count()               = {cpu}')
    print(f'torch.get_num_threads()      = {torch.get_num_threads()}')
    print(f'torch version                = {torch.__version__}')
    print(f'torch.cuda.is_available()    = {torch.cuda.is_available()}')

    print()
    print('Loading Walker2d dataset...')
    t0 = time.time()
    dataset = minari.load_dataset('mujoco/walker2d/medium-v0', download=False)
    print(f'  loaded in {time.time()-t0:.1f}s')
    print(f'  episodes: {dataset.total_episodes}')
    print(f'  steps:    {dataset.total_steps}')

    print()
    print('Building split (seed=1000)...')
    split = deterministic_episode_split(
        dataset.total_episodes, seed=1000,
        train_fraction=0.70, cal_fraction=0.10,
        test_fraction=0.10, probe_fraction=0.10,
    )
    print(f'  train episodes: {len(split["train"])}')
    print(f'  test  episodes: {len(split["test"])}')

    print()
    print('Extracting 150k train transitions (FROZEN cap)...')
    t0 = time.time()
    train_obs, train_act, train_y = extract_transitions(
        dataset, split['train'], max_transitions=150_000, seed=1000)
    print(f'  extracted in {time.time()-t0:.1f}s')
    print(f'  train_obs.shape = {train_obs.shape}')
    print(f'  train_act.shape = {train_act.shape}')

    obs_dim = train_obs.shape[1]
    act_dim = train_act.shape[1]
    print(f'  obs_dim={obs_dim}, act_dim={act_dim}')

    print()
    print('Fitting standardizer + preparing tensors...')
    t0 = time.time()
    std = fit_standardizer(train_obs, train_act, train_y)
    xz, yz = apply_standardizer(std, train_obs, train_act, train_y)
    X = torch.from_numpy(xz)
    Y = torch.from_numpy(yz)
    print(f'  prepared in {time.time()-t0:.1f}s')

    batch_size = 1024
    member_seed = 0

    g = torch.Generator()
    g.manual_seed(member_seed)
    loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=batch_size, shuffle=True, generator=g,
        num_workers=0, pin_memory=False,
    )

    Model = build_model_class()
    model = Model(obs_dim + act_dim, obs_dim, hidden=(256, 256))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    print()
    print(f'Model params: {sum(p.numel() for p in model.parameters())}')
    print(f'Running 2 epochs (batch_size={batch_size})...')

    epoch_times = []
    for epoch in range(1, 3):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            mu, logvar = model(xb)
            loss = gaussian_nll(mu, logvar, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dt = time.time() - t0
        epoch_times.append(dt)
        print(f'  epoch {epoch}: {dt:.1f}s  train_nll={np.mean(losses):.4f}')

    mean_epoch = float(np.mean(epoch_times))
    est_25 = mean_epoch * 25
    print()
    print(f'mean per-epoch time: {mean_epoch:.1f}s')
    print(f'extrapolated to 25 epochs: {est_25:.1f}s = {est_25/60:.1f} min per member')

    n_walker = 25
    n_hopper = 25
    total_seq = (n_walker + n_hopper) * est_25

    print()
    print('=== Estimates ===')
    print(f'  Walker2d, 25 members, sequential:      {n_walker*est_25/3600:.2f} hours')
    print(f'  Hopper 5..9, 25 members, sequential:   {n_hopper*est_25/3600:.2f} hours')
    print(f'  TOTAL sequential:                      {total_seq/3600:.2f} hours')

    if cpu and cpu >= 10:
        print()
        print(f'=== 5-way parallel estimate ({cpu} cores) ===')
        for factor in (3, 4, 5):
            if cpu >= factor * 2:
                par = total_seq / factor
                print(f'  {factor}-way parallel: {par/3600:.2f} hours '
                      f'({factor} procs x ~{cpu//factor} threads each)')

    print()
    print('DECISION GATE:')
    if total_seq / 3600 <= 8.0:
        print('  -> total <= 8h; sequential is acceptable; parallel not required')
    else:
        print('  -> total > 8h; parallel or Colab strongly recommended')

    print()
    print('DONE')


if __name__ == '__main__':
    main()