# [VERIFY] Resolutions — Paper 1 Method

This file resolves the five placeholders that appeared in
`04_method.md`. Each was checked against the actual source code
via `scripts/verify_claims.py`.

## §4.2 — Interval construction

**Answer: Gaussian(mu, total_var).** `predict_ensemble` returns
`mu` and `var_total = mean_k(sigma_k^2) + Var_k(mu_k)`. The
calibration curve builds central intervals as
`mu +/- norm.ppf(1 - alpha/2) * sqrt(var)`. No mixture quantiles.

## §4.3 — Split per mechanism

| Mechanism | Evaluation instances | Baseline | Treatment |
|---|---|---|---|
| Observation (O) | test split (25k) | y_dataset | y_dataset |
| Dynamics (D) | test subsample (5000, seed 11000+idx) | y_resim(mass=1.0) | y_resim(mass=scale) |
| Policy (P) | probe split (10k) | y_resim(action=orig) | y_resim(action=scaled) |

Calibration split is used only to fit `alpha`.

## §4.4 — Observation noise scale

**Answer: per-dim.** `compute_obs_scale` returns
`np.std(test_obs, axis=0)` (shape obs_dim,), applied elementwise.

## §4.6 — Instance CE aggregation

**Answer: dims first.** `inside.mean(axis=1)` -> `abs(inside - c)`
-> `mean` over 9 levels -> shape (N,).

## §4.8 — Bootstrap unit and n_resamples

**Answer: paired instances.** `n_boot = 5000` (O, P), `2000` (D).
Primary CIs in the manuscript are Student-t at seed level.

## Revision history

- 2026-10-06: Resolved all five placeholders via `scripts/verify_claims.py`.
