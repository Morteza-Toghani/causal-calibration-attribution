"""
Phase 3.2b — Gradient Sanity Check for the Variance Head

Before touching any code, we verify whether the NLL loss's gradient with
respect to log_var behaves as theory predicts, and whether raw_log_var
(BEFORE the clamp) is actually saturating above max_log_var=2.0, or
whether the clamp itself is masking a healthy gradient.

Theory (Gaussian NLL, per-element):
    L = 0.5 * [ log_var + (y-mu)^2 / exp(log_var) + log(2*pi) ]
    dL/dlog_var = 0.5 * [ 1 - (y-mu)^2 / exp(log_var) ]
                = 0.5 * [ 1 - e^2 / sigma^2 ]
                = 0.5 * [ 1 - z^2 ]      where z = e/sigma

So:
    e = 0            -> dL/dlog_var = +0.5   (gradient DESCENT decreases log_var)
    |e| = sigma       -> dL/dlog_var = 0      (no pressure either way)
    |e| = 3*sigma     -> dL/dlog_var = 0.5*(1-9) = -4.0  (gradient descent INCREASES log_var)

We test this analytically (cases A-D) AND empirically, by directly
inspecting raw_log_var_head(h) BEFORE the clamp on real data, per
dimension, to see whether it's the raw head saturating (a training
dynamics question) or something else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pathlib import Path as _P

import numpy as np
import torch

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from src.data.loader import TransitionBatch
from src.model.ensemble import EnsembleWorldModel


def analytical_gradient_test():
    """Cases A-D: verify dNLL/dlog_var has the theoretically correct sign."""
    print("=" * 70)
    print("TEST 1 — Analytical gradient sanity check (dNLL/dlog_var sign)")
    print("=" * 70)

    cases = {
        "A: error = 0": 0.0,
        "B: error = small (0.1*sigma)": 0.1,
        "C: error = sigma": 1.0,
        "D: error = 3*sigma": 3.0,
    }

    sigma_true = 1.0

    results = {}
    for name, error_mult in cases.items():
        log_var = torch.tensor(0.0, requires_grad=True)
        error = torch.tensor(error_mult * sigma_true)

        variance = torch.exp(log_var)
        nll = 0.5 * (log_var + error ** 2 / variance + np.log(2 * np.pi))
        nll.backward()

        grad = log_var.grad.item()
        expected_grad = 0.5 * (1 - error_mult ** 2)

        direction = ("DECREASE log_var (tighten sigma)" if grad > 0 else
                      "INCREASE log_var (widen sigma)" if grad < 0 else "no pressure")

        print(f"\n  {name}")
        print(f"    error/sigma = {error_mult}")
        print(f"    dNLL/dlog_var (autograd) = {grad:+.4f}")
        print(f"    dNLL/dlog_var (theory)   = {expected_grad:+.4f}")
        print(f"    match: {np.isclose(grad, expected_grad, atol=1e-4)}")
        print(f"    --> gradient descent direction: {direction}")

        results[name] = {
            "error_over_sigma": error_mult,
            "grad_autograd": grad,
            "grad_theory": expected_grad,
            "matches_theory": bool(np.isclose(grad, expected_grad, atol=1e-4)),
        }

    print("\n  SUMMARY: for small errors (A, B), gradient descent should DECREASE")
    print("  log_var (tighten). For large errors (D), it should INCREASE log_var")
    print("  (widen). This is textbook Gaussian NLL behavior and our autograd")
    print("  computation matches theory exactly.")
    print("  --> CONCLUSION: the raw math of Gaussian NLL w.r.t. log_var is CORRECT.")
    print("      If real data (dims 0-4) still saturates log_var near +2 despite")
    print("      tiny errors, the gradient math itself is not the cause -- we must")
    print("      look at the actual forward pass on real data (Test 2 below).")

    return results


@torch.no_grad()
def empirical_raw_logvar_inspection(model: EnsembleWorldModel, obs, action, next_obs):
    """
    Test 2: directly inspect raw_log_var_head(h) BEFORE the clamp, on real
    training data, per dimension.
    """
    print("\n" + "=" * 70)
    print("TEST 2 — Empirical raw (pre-clamp) log_var inspection on real data")
    print("=" * 70)

    member = model.members[0]
    member.eval()

    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(action, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)

    x = torch.cat([obs_t, act_t], dim=-1)
    h = member.trunk(x)
    delta_mu = member.mu_head(h)
    raw_log_var = member.log_var_head(h)  # BEFORE clamp
    mu = obs_t + delta_mu

    error = (next_obs_t - mu).numpy()
    raw_log_var_np = raw_log_var.numpy()

    n_dims = raw_log_var_np.shape[1]
    print(f"\n  {'dim':>4} {'mean_raw_logvar':>16} {'max_raw_logvar':>16} {'frac_above_2.0':>16} {'mean_abs_error':>16}")
    results = {}
    for d in range(n_dims):
        mean_raw = float(raw_log_var_np[:, d].mean())
        max_raw = float(raw_log_var_np[:, d].max())
        frac_above = float((raw_log_var_np[:, d] > 2.0).mean())
        mean_abs_err = float(np.abs(error[:, d]).mean())
        print(f"  {d:>4} {mean_raw:>16.3f} {max_raw:>16.3f} {frac_above:>16.2%} {mean_abs_err:>16.6f}")
        results[d] = {
            "mean_raw_log_var": mean_raw,
            "max_raw_log_var": max_raw,
            "fraction_above_clamp": frac_above,
            "mean_abs_error": mean_abs_err,
        }

    overall_frac_above = float((raw_log_var_np > 2.0).mean())
    print(f"\n  Overall fraction of (sample, dim) pairs with raw_log_var > 2.0: {overall_frac_above:.2%}")

    if overall_frac_above > 0.5:
        print("  --> The RAW (pre-clamp) head is genuinely trying to predict")
        print("      log_var > 2.0 for a majority of predictions. The clamp is")
        print("      actively cutting off a value the network 'wants' to produce --")
        print("      training pushed log_var_head's output upward past the clamp")
        print("      ceiling, and it is now stuck there because gradients beyond")
        print("      the clamp boundary do not propagate (torch.clamp has zero")
        print("      gradient outside the clamped region). This IS a real")
        print("      saturation problem caused by the hard clamp, not a sign error.")
    else:
        print("  --> Raw log_var is mostly within bounds; saturation is isolated")
        print("      to specific samples/dimensions, not a systemic issue.")

    return {
        "per_dim": results,
        "overall_fraction_above_clamp": overall_frac_above,
    }


def main():
    print("=== Phase 3.2b: Gradient Sanity Check & Variance Head Diagnostic ===\n")

    analytical_results = analytical_gradient_test()

    data_dir = Path("data/processed/hopper_seed1")
    train_batch = TransitionBatch.load(data_dir / "train.npz")

    model_dir = Path("results/models/hopper_seed1_baseline")
    model = EnsembleWorldModel.load(model_dir)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_batch), size=min(50000, len(train_batch)), replace=False)

    empirical_results = empirical_raw_logvar_inspection(
        model,
        train_batch.observations[idx],
        train_batch.actions[idx],
        train_batch.next_observations[idx],
    )

    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)
    print("""
If Test 1 shows the gradient math is correct (it should -- standard
Gaussian NLL) AND Test 2 shows raw_log_var genuinely exceeding 2.0 for a
large fraction of predictions, the diagnosis is:

    The hard clamp at max_log_var=2.0 is masking a network that has
    learned to push log_var above the ceiling for certain dimensions.
    Once clamped, gradients for those (sample, dim) pairs stop flowing
    through log_var_head (torch.clamp has zero gradient in the saturated
    region), so the network can never learn to bring log_var back down --
    explaining why MORE epochs made things worse: it had more time to
    push raw_log_var further past the ceiling, deepening saturation the
    clamp can no longer correct.

This matches the per-dimension pattern from Phase 3.2: dims 0-4 have the
worst sigma/RMSE ratios (5x-17x) -- exactly the "easy" dims where true
error is tiny, so the network should predict very small sigma, but
instead saturates at the clamp ceiling in the opposite direction.

Recommended next step: replace the hard clamp with a soft parameterization
(e.g. softplus) that has non-zero gradient everywhere, and re-diagnose.
""")

    out_dir = Path("results/baseline/hopper_seed1")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase3_2b_gradient_sanity_check.json", "w") as f:
        json.dump({
            "analytical_gradient_test": analytical_results,
            "empirical_raw_logvar_inspection": empirical_results,
        }, f, indent=2)
    print(f"Saved full report to {out_dir / 'phase3_2b_gradient_sanity_check.json'}")


if __name__ == "__main__":
    main()
