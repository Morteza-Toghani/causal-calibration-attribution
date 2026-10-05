"""
Phase 2 — Probabilistic MLP Ensemble World Model

Implements the core world model described in the Master Research Plan
(Section 6): a K=5 member ensemble of MLPs, each predicting a diagonal
Gaussian distribution over the next state s_{t+1} given (s_t, a_t):

    p_k(s_{t+1} | s_t, a_t) = Normal(mu_k, sigma_k^2)

Ensemble predictive mean and total (predictive) variance are combined as:

    mu          = mean_k(mu_k)
    Var_total   = mean_k(sigma_k^2) + Var_k(mu_k)
                  (aleatoric)          (epistemic / disagreement)

This matches the aleatoric/epistemic decomposition used by deep ensembles
(Lakshminarayanan et al. 2017) referenced in the master plan.

Each member is trained independently with a Gaussian negative log-likelihood
loss (NLL), which is the standard way to train a heteroscedastic Gaussian
head (Nix & Weigend style, used by Lakshminarayanan et al.).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Single probabilistic MLP member
# ---------------------------------------------------------------------------

class GaussianMLP(nn.Module):
    """
    A single ensemble member: MLP mapping (s_t, a_t) -> (mu, log_sigma2)
    for predicting s_{t+1}.

    log_sigma2 is predicted (rather than sigma2 directly) for numerical
    stability, and clamped to a reasonable range to avoid collapse/explosion
    early in training.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (200, 200, 200),
        min_log_var: float = -20.0,
        max_log_var: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        input_dim = obs_dim + action_dim
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Predict delta (s_{t+1} - s_t) rather than raw s_{t+1}: standard
        # practice for MLP dynamics models, makes the regression target
        # smaller-scale and easier to fit.
        self.mu_head = nn.Linear(prev, obs_dim)
        self.log_var_head = nn.Linear(prev, obs_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        h = self.trunk(x)
        delta_mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)

        mu = obs + delta_mu  # predict next state directly via residual
        sigma2 = torch.exp(log_var)
        return mu, sigma2

    def nll_loss(self, obs, action, next_obs) -> torch.Tensor:
        mu, sigma2 = self.forward(obs, action)
        # Gaussian NLL, per-dimension, averaged over batch and dims
        nll = 0.5 * (torch.log(sigma2) + (next_obs - mu) ** 2 / sigma2)
        return nll.mean()


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePrediction:
    mu: np.ndarray            # (N, obs_dim) predictive mean
    var_total: np.ndarray     # (N, obs_dim) total predictive variance
    var_aleatoric: np.ndarray  # (N, obs_dim) mean of member variances
    var_epistemic: np.ndarray  # (N, obs_dim) variance of member means
    member_mus: np.ndarray    # (K, N, obs_dim)
    member_sigma2s: np.ndarray  # (K, N, obs_dim)


class EnsembleWorldModel:
    """K-member ensemble of GaussianMLP, trained independently."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_members: int = 5,
        hidden_sizes: tuple[int, ...] = (200, 200, 200),
        seed: int = 1,
        device: str = "cpu",
        *,
        min_log_var: float,
        max_log_var: float = 2.0,
    ):
        self.hidden_sizes = tuple(hidden_sizes)
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_members = n_members
        self.device = device

        self.members: list[GaussianMLP] = []
        for k in range(n_members):
            torch.manual_seed(seed * 1000 + k)  # distinct init per member
            member = GaussianMLP(
                obs_dim, action_dim, hidden_sizes,
                min_log_var=self.min_log_var,
                max_log_var=self.max_log_var,
            ).to(device)
            self.members.append(member)

    def train_member(
        self,
        member_idx: int,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        obs_val: np.ndarray | None = None,
        action_val: np.ndarray | None = None,
        next_obs_val: np.ndarray | None = None,
        n_epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 1,
        log_every: int = 5,
    ) -> dict:
        """Train a single ensemble member with Adam + Gaussian NLL loss."""
        member = self.members[member_idx]
        opt = torch.optim.Adam(member.parameters(), lr=lr)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        n = obs_t.shape[0]
        rng = np.random.default_rng(seed * 1000 + member_idx)

        history = {"train_nll": [], "val_nll": []}

        for epoch in range(n_epochs):
            perm = rng.permutation(n)
            epoch_losses = []
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                b_obs = obs_t[idx]
                b_act = act_t[idx]
                b_next = next_t[idx]

                opt.zero_grad()
                loss = member.nll_loss(b_obs, b_act, b_next)
                loss.backward()
                # gradient clipping for stability (per master plan risk
                # mitigation: "gradient clipping, NLL monitoring")
                torch.nn.utils.clip_grad_norm_(member.parameters(), max_norm=5.0)
                opt.step()
                epoch_losses.append(loss.item())

            mean_train_nll = float(np.mean(epoch_losses))
            history["train_nll"].append(mean_train_nll)

            val_nll = None
            if obs_val is not None:
                with torch.no_grad():
                    val_nll = member.nll_loss(
                        torch.as_tensor(obs_val, dtype=torch.float32, device=self.device),
                        torch.as_tensor(action_val, dtype=torch.float32, device=self.device),
                        torch.as_tensor(next_obs_val, dtype=torch.float32, device=self.device),
                    ).item()
                history["val_nll"].append(val_nll)

            if (epoch + 1) % log_every == 0 or epoch == 0:
                msg = f"  member {member_idx} epoch {epoch + 1:>3}/{n_epochs}  train_nll={mean_train_nll:.4f}"
                if val_nll is not None:
                    msg += f"  val_nll={val_nll:.4f}"
                print(msg)

        return history

    def train_all(
        self,
        obs, action, next_obs,
        obs_val=None, action_val=None, next_obs_val=None,
        n_epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 1,
    ) -> dict:
        all_history = {}
        for k in range(self.n_members):
            print(f"\n--- Training ensemble member {k+1}/{self.n_members} ---")
            all_history[k] = self.train_member(
                k, obs, action, next_obs,
                obs_val, action_val, next_obs_val,
                n_epochs=n_epochs, batch_size=batch_size, lr=lr, seed=seed,
            )
        return all_history

    @torch.no_grad()
    def predict(self, obs: np.ndarray, action: np.ndarray) -> EnsemblePrediction:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        mus, sigma2s = [], []
        for member in self.members:
            member.eval()
            mu, sigma2 = member(obs_t, act_t)
            mus.append(mu.cpu().numpy())
            sigma2s.append(sigma2.cpu().numpy())

        mus = np.stack(mus, axis=0)         # (K, N, obs_dim)
        sigma2s = np.stack(sigma2s, axis=0)  # (K, N, obs_dim)

        mu = mus.mean(axis=0)
        var_aleatoric = sigma2s.mean(axis=0)
        var_epistemic = mus.var(axis=0)
        var_total = var_aleatoric + var_epistemic

        return EnsemblePrediction(
            mu=mu,
            var_total=var_total,
            var_aleatoric=var_aleatoric,
            var_epistemic=var_epistemic,
            member_mus=mus,
            member_sigma2s=sigma2s,
        )

    def save(self, path, extra_meta=None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for k, member in enumerate(self.members):
            torch.save(member.state_dict(), path / f"member_{k}.pt")
        meta = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "n_members": self.n_members,
            "hidden_sizes": list(self.hidden_sizes),
            "min_log_var": self.min_log_var,
            "max_log_var": self.max_log_var,
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "EnsembleWorldModel":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        for key in ("min_log_var", "max_log_var"):
            if key not in meta:
                raise KeyError(f"meta.json lacks {key}")
        hs = tuple(meta.get("hidden_sizes", (200, 200, 200)))
        model = cls(
            hidden_sizes=hs,
            min_log_var=meta["min_log_var"],
            max_log_var=meta["max_log_var"],
            obs_dim=meta["obs_dim"],
            action_dim=meta["action_dim"],
            n_members=meta["n_members"],
            device=device,
        )
        for k, member in enumerate(model.members):
            state = torch.load(path / f"member_{k}.pt", map_location=device)
            member.load_state_dict(state)
        return model


# ---------------------------------------------------------------------------
# Script entry point: train baseline ensemble on the Phase 1 pilot split
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.loader import TransitionBatch

    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--min-log-var", type=float, required=True)
    ap.add_argument("--max-log-var", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--data-dir", default="data/processed/hopper_seed1"
    )
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; not overwriting")

    d = Path(a.data_dir)
    tr = TransitionBatch.load(d / "train.npz")
    ca = TransitionBatch.load(d / "calibration.npz")

    model = EnsembleWorldModel(
        obs_dim=tr.observations.shape[1],
        action_dim=tr.actions.shape[1],
        n_members=5,
        seed=a.seed,
        min_log_var=a.min_log_var,
        max_log_var=a.max_log_var,
    )
    history = model.train_all(
        tr.observations, tr.actions, tr.next_observations,
        ca.observations, ca.actions, ca.next_observations,
        n_epochs=a.epochs, batch_size=256, lr=1e-3, seed=a.seed,
    )
    model.save(out_dir, extra_meta={
        "config_id": a.config_id,
        "training_seed": a.seed,
        "epochs": a.epochs,
    })
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved to", out_dir)
