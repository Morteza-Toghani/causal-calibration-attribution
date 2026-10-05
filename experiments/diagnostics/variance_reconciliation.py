"""
Phase 3.2c — Raw/Effective/Predictive Variance Reconciliation

Resolves the apparent mismatch between:
  - raw_log_var range seen in gradient_sanity_check.py (e.g. [-20, -7.7] for dim 0)
  - mean_sigma=0.00679 seen in variance_diagnostic.py for the same dimension

by computing EVERYTHING from a single forward pass on the same data, with
raw_log_var, effective_log_var (post-clamp), aleatoric variance/sigma, and
total predictive variance/sigma all reported side by side, per dimension.

Also explicitly checks LOWER clamp saturation (min_log_var=-10.0), which
was not checked in the previous (upper-clamp-only) diagnostic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pathlib import Path as _P

import numpy as np
import torch

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from src.calibration.regression import ensemble_moments
from src.data.loader import TransitionBatch
from src.model.ensemble import EnsembleWorldModel


@torch.no_grad()
def reconcile(model: EnsembleWorldModel, obs, action, next_obs):
    n_dims = obs.shape[1]
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(action, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)

    member = model.members[0]
    member.eval()

    # --- Manual trace for member 0 ---
    x = torch.cat([obs_t, act_t], dim=-1)
    h = member.trunk(x)
    delta_mu = member.mu_head(h)
    raw_log_var = member.log_var_head(h)                          # BEFORE clamp
    effective_log_var = torch.clamp(
        raw_log_var, member.min_log_var, member.max_log_var
    )                                                              # AFTER clamp
    mu = obs_t + delta_mu
    aleatoric_variance = torch.exp(effective_log_var)              # member 0's sigma_0^2
    aleatoric_sigma = torch.sqrt(aleatoric_variance)

    error = (next_obs_t - mu).numpy()

    # --- Full ensemble prediction (all 5 members) for total predictive variance ---
    member_means, member_vars = [], []
    for m in model.members:
        m.eval()
        mu_k, sigma2_k = m(obs_t, act_t)
        member_means.append(mu_k.numpy())
        member_vars.append(sigma2_k.numpy())
    member_means = np.stack(member_means, axis=0)
    member_vars = np.stack(member_vars, axis=0)
    predictive_mean, predictive_variance = ensemble_moments(member_means, member_vars)
    predictive_sigma = np.sqrt(predictive_variance)

    raw_log_var_np = raw_log_var.numpy()
    effective_log_var_np = effective_log_var.numpy()
    aleatoric_variance_np = aleatoric_variance.numpy()
    aleatoric_sigma_np = aleatoric_sigma.numpy()

    tolerance = 0.5  # "near the bound" tolerance in log_var units
    lower_bound = member.min_log_var
    upper_bound = member.max_log_var

    print(f"{'dim':>4} | {'raw_logvar':>26} | {'effective_logvar':>26} | {'aleatoric_sigma (mem0)':>24} | {'predictive_sigma (all)':>24} | {'RMSE':>10} | {'sat_lo%':>8} | {'sat_hi%':>8}")
    print("-" * 175)

    results = {}
    for d in range(n_dims):
        raw = raw_log_var_np[:, d]
        eff = effective_log_var_np[:, d]
        al_sigma = aleatoric_sigma_np[:, d]
        pred_sigma = predictive_sigma[:, d]
        err_d = error[:, d]

        rmse = float(np.sqrt((err_d ** 2).mean()))
        mean_abs_err = float(np.abs(err_d).mean())

        frac_lower_sat = float((raw <= lower_bound + tolerance).mean())
        frac_upper_sat = float((raw >= upper_bound - tolerance).mean())

        raw_str = f"[{raw.min():.2f},{raw.max():.2f}] m={raw.mean():.2f} md={np.median(raw):.2f}"
        eff_str = f"[{eff.min():.2f},{eff.max():.2f}] m={eff.mean():.2f} md={np.median(eff):.2f}"
        al_str = f"[{al_sigma.min():.5f},{al_sigma.max():.5f}] m={al_sigma.mean():.5f}"
        pred_str = f"[{pred_sigma.min():.5f},{pred_sigma.max():.5f}] m={pred_sigma.mean():.5f}"

        print(f"{d:>4} | {raw_str:>26} | {eff_str:>26} | {al_str:>24} | {pred_str:>24} | {rmse:>10.6f} | {frac_lower_sat*100:>7.2f}% | {frac_upper_sat*100:>7.2f}%")

        # The four-number determinative test requested:
        mean_raw = float(raw.mean())
        median_raw = float(np.median(raw))
        mean_al_sigma = float(al_sigma.mean())
        median_al_sigma = float(np.median(al_sigma))
        exp_mean_raw_half = float(np.exp(mean_raw / 2))
        sqrt_mean_aleatoric_var = float(np.sqrt(aleatoric_variance_np[:, d].mean()))

        results[d] = {
            "raw_log_var": {"min": float(raw.min()), "max": float(raw.max()),
                             "mean": mean_raw, "median": median_raw},
            "effective_log_var": {"min": float(eff.min()), "max": float(eff.max()),
                                   "mean": float(eff.mean()), "median": float(np.median(eff))},
            "aleatoric_sigma_member0": {"min": float(al_sigma.min()), "max": float(al_sigma.max()),
                                         "mean": mean_al_sigma, "median": median_al_sigma},
            "predictive_sigma_ensemble": {"min": float(pred_sigma.min()), "max": float(pred_sigma.max()),
                                           "mean": float(pred_sigma.mean())},
            "rmse": rmse,
            "mean_abs_error": mean_abs_err,
            "sigma_over_mean_abs_error_predictive": float(pred_sigma.mean() / max(mean_abs_err, 1e-12)),
            "fraction_lower_saturated": frac_lower_sat,
            "fraction_upper_saturated": frac_upper_sat,
            # The four-number reconciliation test:
            "exp_of_mean_raw_logvar_over_2": exp_mean_raw_half,
            "sqrt_of_mean_aleatoric_variance": sqrt_mean_aleatoric_var,
            "note": "exp(mean(x)/2) != sqrt(mean(exp(x))) by Jensen's inequality -- "
                    "these are expected to differ; that is not a bug.",
        }

    return results


def main():
    print("=== Phase 3.2c: Raw / Effective / Predictive Variance Reconciliation ===")
    print("(Hopper, seed=1, 50-epoch model, member 0 raw trace + full 5-member ensemble)\n")

    data_dir = Path("data/processed/hopper_seed1")
    train_batch = TransitionBatch.load(data_dir / "train.npz")

    model_dir = Path("results/models/hopper_seed1_baseline")
    model = EnsembleWorldModel.load(model_dir)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_batch), size=min(50000, len(train_batch)), replace=False)

    results = reconcile(
        model,
        train_batch.observations[idx],
        train_batch.actions[idx],
        train_batch.next_observations[idx],
    )

    print(f"\nClamp bounds in effect: min_log_var={model.members[0].min_log_var}, "
          f"max_log_var={model.members[0].max_log_var}")

    print("\n" + "=" * 70)
    print("FOUR-NUMBER DETERMINATIVE TEST (dim 0, as requested)")
    print("=" * 70)
    d0 = results[0]
    print(f"  mean(raw_log_var)                : {d0['raw_log_var']['mean']:.4f}")
    print(f"  median(raw_log_var)               : {d0['raw_log_var']['median']:.4f}")
    print(f"  mean(aleatoric_sigma, member 0)   : {d0['aleatoric_sigma_member0']['mean']:.6f}")
    print(f"  median(aleatoric_sigma, member 0) : {d0['aleatoric_sigma_member0']['median']:.6f}")
    print(f"  exp(mean(raw_log_var)/2)          : {d0['exp_of_mean_raw_logvar_over_2']:.6f}")
    print(f"  sqrt(mean(aleatoric_variance))    : {d0['sqrt_of_mean_aleatoric_variance']:.6f}")
    print()
    print("  Note: exp(mean(log_var)/2) and sqrt(mean(exp(log_var))) are NOT")
    print("  mathematically required to match (Jensen's inequality: E[exp(X)] >=")
    print("  exp(E[X]) for convex exp). A gap between these two numbers is NOT a")
    print("  bug -- it's expected whenever log_var has any spread across samples.")

    print("\n" + "=" * 70)
    print("LOWER-CLAMP SATURATION CHECK (previously unchecked)")
    print("=" * 70)
    for d in range(11):
        r = results[d]
        flag = "  <-- SATURATING AT LOWER BOUND" if r["fraction_lower_saturated"] > 0.05 else ""
        print(f"  dim {d:2d}: lower-sat={r['fraction_lower_saturated']*100:6.2f}%  "
              f"upper-sat={r['fraction_upper_saturated']*100:6.2f}%{flag}")

    out_dir = Path("results/baseline/hopper_seed1")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase3_2c_variance_reconciliation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full reconciliation report to {out_dir / 'phase3_2c_variance_reconciliation.json'}")


if __name__ == "__main__":
    main()
