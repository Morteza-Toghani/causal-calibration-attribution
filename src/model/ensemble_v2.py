"""Paper 1 v2 ensemble model — matches FROZEN pipeline exactly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.model.standardizer import Standardizer, apply_standardizer


class ProbabilisticMLP(nn.Module):
    """Matches FROZEN pipeline: ReLU backbone + clamp[-10, 5] on logvar."""

    def __init__(self, input_dim: int, output_dim: int, hidden=(256, 256)):
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


@dataclass
class LoadedMember:
    model: ProbabilisticMLP
    standardizer: Standardizer


def load_member(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden = tuple(ckpt['hidden'])
    model = ProbabilisticMLP(ckpt['input_dim'], ckpt['output_dim'], hidden).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    std = Standardizer(**ckpt['standardizer'])
    return LoadedMember(model=model, standardizer=std)


def predict_ensemble(members, obs, act, device='cpu',
                     observation_noise=None, action_transform=None):
    obs2 = obs.copy()
    act2 = act.copy()

    if observation_noise is not None:
        obs2 = obs2 + observation_noise.astype(np.float32)
    if action_transform is not None:
        act2 = action_transform(act2).astype(np.float32)

    outputs_mu = []
    outputs_var = []

    for item in members:
        std = item.standardizer
        xz = apply_standardizer(std, obs2, act2)
        xt = torch.from_numpy(xz).to(device)

        model = item.model
        model.eval()
        with torch.no_grad():
            mu_z, logvar_z = model(xt)
            mu_z = mu_z.cpu().numpy()
            var_z = torch.exp(logvar_z).cpu().numpy()

        my = np.asarray(std.mean_y, dtype=np.float32)
        sy = np.asarray(std.std_y, dtype=np.float32)

        mu = mu_z * sy + my
        var = var_z * (sy ** 2)

        outputs_mu.append(mu)
        outputs_var.append(var)

    member_mu = np.stack(outputs_mu, axis=0)
    member_var = np.stack(outputs_var, axis=0)

    predictive_mu = member_mu.mean(axis=0)
    predictive_var = member_var.mean(axis=0) + member_mu.var(axis=0)

    return {
        'member_mu': member_mu,
        'member_var': member_var,
        'mu': predictive_mu,
        'var': np.maximum(predictive_var, 1e-9),
    }
