"""
Phase 3 — Evaluate baseline calibration on the held-out test set.

Uses src/calibration/regression.py (the frozen Phase 3 metric contract:
regression calibration error at 50%/90% nominal coverage, NLL, CRPS,
per-dimension + macro-averaged) applied to the Phase 2 baseline 5-member
ensemble, evaluated ONLY on the test split (never on calibration/probe).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from src.data.loader import TransitionBatch
from src.model.ensemble import EnsembleWorldModel
from src.calibration.regression import evaluate_regression_calibration


@torch.no_grad()
def get_member_moments(model: EnsembleWorldModel, obs: np.ndarray, action: np.ndarray):
    """
    Run each ensemble member individually and return raw per-member
    (mu_k, sigma_k^2) with shape [K, N, D], as required by
    evaluate_regression_calibration.
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    act_t = torch.as_tensor(action, dtype=torch.float32, device=model.device)

    member_means, member_vars = [], []
    for member in model.members:
        member.eval()
        mu, sigma2 = member(obs_t, act_t)
        member_means.append(mu.cpu().numpy())
        member_vars.append(sigma2.cpu().numpy())

    return np.stack(member_means, axis=0), np.stack(member_vars, axis=0)


def main():
    print("=== Phase 3: Baseline regression calibration on held-out TEST set ===\n")
    print("(evaluation split: TEST only; calibration split reserved for future use)\n")

    data_dir = Path("data/processed/hopper_seed1")
    test_batch = TransitionBatch.load(data_dir / "test.npz")

    model_dir = Path("results/models/hopper_seed1_baseline")
    model = EnsembleWorldModel.load(model_dir)

    member_means, member_vars = get_member_moments(
        model, test_batch.observations, test_batch.actions
    )

    result = evaluate_regression_calibration(
        y_true=test_batch.next_observations,
        member_means=member_means,
        member_variances=member_vars,
        nominal_levels=(0.50, 0.90),
    )

    report = {
        "environment": "Hopper",
        "dataset": "mujoco/hopper/medium-v0",
        "training_seed": 1,
        "ensemble_members": model.n_members,
        "evaluation_split": "test",
        "prediction_horizon": 1,
        "nominal_levels": [0.5, 0.9],
        **result,
    }

    print(f"n_samples             : {report['n_samples']}")
    print(f"n_dimensions           : {report['n_dimensions']}")
    print(f"calibration_error      : {report['calibration_error']:.6f}")
    print(f"nll_mean                : {report['nll_mean']:.6f}")
    print(f"crps_mean               : {report['crps_mean']:.6f}")
    print(f"predictive_variance_mean: {report['predictive_variance_mean']:.6f}")
    for level_str, level_data in report["calibration"]["levels"].items():
        print(
            f"  level {level_str}: empirical_coverage="
            f"{level_data['empirical_coverage_mean']:.4f} "
            f"(error {level_data['coverage_error_mean']:+.4f}), "
            f"sharpness={level_data['sharpness_mean']:.4f}"
        )

    out_dir = Path("results/baseline/hopper_seed1")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # Also write a simple per-dimension CSV for quick inspection
    import csv
    n_dims = report["n_dimensions"]
    with open(out_dir / "per_dimension_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dimension", "nll", "crps",
            "coverage_50", "coverage_error_50",
            "coverage_90", "coverage_error_90",
        ])
        cov50 = report["calibration"]["levels"]["0.5"]
        cov90 = report["calibration"]["levels"]["0.9"]
        for d in range(n_dims):
            writer.writerow([
                d,
                report["nll_per_dimension"][d],
                report["crps_per_dimension"][d],
                cov50["coverage_per_dimension"][d],
                cov50["coverage_error_per_dimension"][d],
                cov90["coverage_per_dimension"][d],
                cov90["coverage_error_per_dimension"][d],
            ])

    print(f"\nSaved: {out_dir/'metrics.json'}")
    print(f"Saved: {out_dir/'per_dimension_metrics.csv'}")


if __name__ == "__main__":
    main()
