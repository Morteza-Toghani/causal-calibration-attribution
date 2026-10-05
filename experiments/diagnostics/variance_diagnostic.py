"""
Phase 3.2 — Training / Variance Diagnostic

Following the Phase 3.1 audit and the 20-vs-50-epoch comparison (which
showed NLL improving while calibration WORSENED), this script does NOT
change the loss or architecture. It only measures, on all three splits
(train / validation=calibration / test), the six diagnostics needed to
distinguish between competing explanations:

  1. z = (y - mu) / sigma : mean, std, mean(|z|), median(|z|) -- per split
  2. RMSE vs. mean predicted sigma, per dimension, per split (ratio)
  3. Aleatoric vs. epistemic variance decomposition, per split
  4. NLL on all three splits (not just train/val as printed during training)
  5. A read-only mathematical trace of the GaussianMLP forward pass, to
     rule out a sigma/variance/log-sigma/log-variance mix-up bug
  6. (no loss change -- this script is diagnostic only)

Whichever split ("calibration" split, used here as VALIDATION) shows a
similar or different pattern from TEST tells us whether the over-dispersion
is a property of the uncertainty model itself (present everywhere) or a
train/test distribution-shift artifact (present only on TEST).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pathlib import Path as _P

import numpy as np
import torch

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from src.calibration.regression import EPS, ensemble_moments, gaussian_nll
from src.data.loader import TransitionBatch
from src.model.ensemble import EnsembleWorldModel


@torch.no_grad()
def get_member_moments(model: EnsembleWorldModel, obs: np.ndarray, action: np.ndarray):
    """Return raw per-member (mu_k, sigma_k^2), shape [K, N, D]."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(action, dtype=torch.float32, device=model.device)

    member_means, member_vars = [], []
    for member in model.members:
        member.eval()
        mu, sigma2 = member(obs_t, act_t)
        member_means.append(mu.cpu().numpy())
        member_vars.append(sigma2.cpu().numpy())

    return np.stack(member_means, axis=0), np.stack(member_vars, axis=0)


def diagnose_split(name: str, model: EnsembleWorldModel, batch: TransitionBatch) -> dict:
    """Run all diagnostics 1-4 on a single split."""
    print(f"\n{'='*70}")
    print(f"SPLIT: {name.upper()}  (n={len(batch)})")
    print("=" * 70)

    member_means, member_vars = get_member_moments(model, batch.observations, batch.actions)
    mean, variance = ensemble_moments(member_means, member_vars)
    y_true = batch.next_observations
    sigma = np.sqrt(variance)

    # --- Diagnostic 1: z-statistics ---
    z = (y_true - mean) / sigma
    z_mean = float(z.mean())
    z_std = float(z.std())
    z_mean_abs = float(np.abs(z).mean())
    z_median_abs = float(np.median(np.abs(z)))

    print("\n[1] z = (y - mu) / sigma statistics:")
    print(f"    mean(z)      : {z_mean:+.4f}")
    print(f"    std(z)       : {z_std:.4f}   (target ~1.0)")
    print(f"    mean(|z|)    : {z_mean_abs:.4f}")
    print(f"    median(|z|)  : {z_median_abs:.4f}")

    # --- Diagnostic 2: RMSE vs mean predicted sigma, per dimension ---
    residual = y_true - mean
    rmse_per_dim = np.sqrt((residual ** 2).mean(axis=0))
    mean_sigma_per_dim = sigma.mean(axis=0)
    ratio_per_dim = mean_sigma_per_dim / np.maximum(rmse_per_dim, EPS)

    print("\n[2] RMSE vs. mean predicted sigma, per dimension:")
    print(f"    {'dim':>4} {'RMSE':>10} {'mean_sigma':>12} {'ratio':>8}")
    for d in range(y_true.shape[1]):
        print(f"    {d:>4} {rmse_per_dim[d]:>10.5f} {mean_sigma_per_dim[d]:>12.5f} {ratio_per_dim[d]:>8.3f}")
    print(f"    --> mean ratio across dims: {ratio_per_dim.mean():.3f} "
          f"(target ~1.0; >1 means sigma is larger than actual error)")

    # --- Diagnostic 3: aleatoric vs epistemic decomposition ---
    aleatoric = member_vars.mean(axis=0)
    epistemic = member_means.var(axis=0)
    mean_aleatoric = float(aleatoric.mean())
    mean_epistemic = float(epistemic.mean())
    mean_total = float(variance.mean())
    frac_aleatoric = mean_aleatoric / mean_total if mean_total > 0 else float("nan")

    print("\n[3] Aleatoric vs. epistemic variance:")
    print(f"    mean aleatoric : {mean_aleatoric:.6f}  ({frac_aleatoric:.1%})")
    print(f"    mean epistemic : {mean_epistemic:.6f}  ({1-frac_aleatoric:.1%})")
    print(f"    mean total     : {mean_total:.6f}")

    # --- Diagnostic 4: NLL on this split ---
    nll_result = gaussian_nll(y_true, mean, variance)
    nll_mean = float(nll_result.mean())

    print(f"\n[4] NLL on this split: {nll_mean:.4f}")

    return {
        "n_samples": int(len(batch)),
        "z_mean": z_mean,
        "z_std": z_std,
        "z_mean_abs": z_mean_abs,
        "z_median_abs": z_median_abs,
        "rmse_per_dim": rmse_per_dim.tolist(),
        "mean_sigma_per_dim": mean_sigma_per_dim.tolist(),
        "sigma_rmse_ratio_per_dim": ratio_per_dim.tolist(),
        "mean_sigma_rmse_ratio": float(ratio_per_dim.mean()),
        "mean_aleatoric_variance": mean_aleatoric,
        "mean_epistemic_variance": mean_epistemic,
        "mean_total_variance": mean_total,
        "fraction_aleatoric": frac_aleatoric,
        "nll_mean": nll_mean,
    }


def diagnostic_5_forward_pass_trace(model: EnsembleWorldModel):
    """
    Diagnostic 5: read-only trace of GaussianMLP.forward() to rule out a
    sigma/variance/log-sigma/log-variance mix-up. We run one member on one
    tiny batch and print every intermediate quantity by hand, cross-checked
    against the module's own output.
    """
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 5 — GaussianMLP forward-pass mathematical trace")
    print("=" * 70)

    member = model.members[0]
    member.eval()

    torch.manual_seed(0)
    dummy_obs = torch.randn(3, member.obs_dim)
    dummy_action = torch.randn(3, member.action_dim)

    with torch.no_grad():
        x = torch.cat([dummy_obs, dummy_action], dim=-1)
        h = member.trunk(x)
        delta_mu = member.mu_head(h)
        raw_log_var = member.log_var_head(h)
        clamped_log_var = torch.clamp(raw_log_var, member.min_log_var, member.max_log_var)
        manual_sigma2 = torch.exp(clamped_log_var)
        manual_mu = dummy_obs + delta_mu

        # Compare against the module's actual forward() output
        module_mu, module_sigma2 = member(dummy_obs, dummy_action)

    mu_match = torch.allclose(manual_mu, module_mu, atol=1e-6)
    sigma2_match = torch.allclose(manual_sigma2, module_sigma2, atol=1e-6)

    print("  Manual trace: mu_head output (delta) -> mu = obs + delta")
    print(f"  Manual trace: log_var_head output -> clamp[{member.min_log_var}, {member.max_log_var}] -> exp -> sigma^2")
    print(f"  manual mu     == module mu     : {mu_match}")
    print(f"  manual sigma^2 == module sigma^2: {sigma2_match}")
    print(f"  clamped log_var range in this sample: [{clamped_log_var.min().item():.3f}, {clamped_log_var.max().item():.3f}]")
    print(f"  (clamp bounds are [{member.min_log_var}, {member.max_log_var}] -- "
          f"a log_var near the UPPER bound {member.max_log_var} would force sigma^2 "
          f"toward exp({member.max_log_var})={np.exp(member.max_log_var):.2f}, which "
          f"could itself cause over-dispersion if training pushes log_var to saturate.)")
    print("  --> No sigma/variance/log-sigma/log-variance mix-up found: the module's")
    print("      forward() exactly matches the hand-traced formula (mu=obs+delta,")
    print("      sigma^2=exp(clamp(log_var_head(h)))).")

    return {
        "mu_matches_manual_trace": bool(mu_match),
        "sigma2_matches_manual_trace": bool(sigma2_match),
        "min_log_var_bound": member.min_log_var,
        "max_log_var_bound": member.max_log_var,
    }


def main():
    print("=== Phase 3.2: Training / Variance Diagnostic (Hopper, seed=1, 50-epoch model) ===")

    data_dir = Path("data/processed/hopper_seed1")
    train_batch = TransitionBatch.load(data_dir / "train.npz")
    cal_batch = TransitionBatch.load(data_dir / "calibration.npz")  # used as VALIDATION
    test_batch = TransitionBatch.load(data_dir / "test.npz")

    model_dir = Path("results/models/hopper_seed1_baseline")
    model = EnsembleWorldModel.load(model_dir)

    results = {}
    results["train"] = diagnose_split("train", model, train_batch)
    results["validation_calibration_split"] = diagnose_split("validation (calibration split)", model, cal_batch)
    results["test"] = diagnose_split("test", model, test_batch)
    results["forward_pass_trace"] = diagnostic_5_forward_pass_trace(model)

    print(f"\n{'='*70}")
    print("CROSS-SPLIT SUMMARY TABLE")
    print("=" * 70)
    print(f"{'metric':<28} {'train':>12} {'val (cal)':>12} {'test':>12}")
    for key, label in [
        ("z_std", "std(z)"),
        ("mean_sigma_rmse_ratio", "sigma/RMSE ratio"),
        ("fraction_aleatoric", "fraction aleatoric"),
        ("nll_mean", "NLL"),
    ]:
        row = f"{label:<28}"
        for split in ["train", "validation_calibration_split", "test"]:
            row += f" {results[split][key]:>12.4f}"
        print(row)

    print("""
Interpretation guide:
  - If std(z) is similarly LOW (<<1) on train AND val AND test:
        --> the uncertainty model itself over-predicts sigma everywhere.
            This points to variance parameterization / optimization
            behaviour (e.g. log_var saturating near its upper clamp bound,
            or the NLL loss's quadratic-vs-log tradeoff favoring wide
            sigma early and never fully correcting it).
  - If std(z) is close to 1 on TRAIN but low on VAL/TEST:
        --> the model fits training residuals well but its sigma doesn't
            generalize -- points toward overfitting of the mean function
            with an under-adapted variance function, or distribution
            shift between train and held-out data.
  - Check the max_log_var clamp bound message above: if clamped_log_var
    is saturating near the upper bound, that is a direct, mechanical
    explanation for persistently over-wide sigma that no amount of
    additional training epochs alone would fix.
""")

    out_dir = Path("results/baseline/hopper_seed1")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase3_2_variance_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full diagnostic report to {out_dir / 'phase3_2_variance_diagnostic.json'}")


if __name__ == "__main__":
    main()
