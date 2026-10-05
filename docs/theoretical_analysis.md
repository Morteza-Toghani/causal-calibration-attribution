# Theoretical analysis (definitions and elementary derivations only)

No formal theorems or proofs are claimed here.

## 1. Mixture moments of an ensemble
For equal-weight members `N(mu_k, sigma_k^2)`: mixture mean `mean_k mu_k`, mixture variance
`mean_k sigma_k^2 + Var_k(mu_k)` (law of total variance, population variance over members). Checked
numerically in `tests/test_models.py`.

## 2. Central-interval coverage
For a Gaussian prediction, the interval `mu +/- z sigma`, `z = Phi^{-1}((1+p)/2)`, has coverage `p` iff the
predictive distribution is calibrated at level `p`. If the predicted sigma is `c` times the true residual
scale, coverage is `2 Phi(z c) - 1`: over-coverage for `c > 1`, under-coverage for `c < 1`.

## 3. The primary metric is unsigned
`calibration_error = mean_p |coverage(p) - p|` discards the sign. A shift that moves an over-covering model
(`c > 1`) past nominal into under-coverage (`c < 1`) can *lower* the metric. The Phase-B data show this
case: observation low has a smaller calibration error than baseline while its sign of bias is reversed.
Hence signed bias is reported alongside (`src/calibration/scalar_metrics.py`). Pooling coverage over dimensions
before the absolute value can additionally mask opposite-sign dimensions (`calibration_error_macro`).

## 4. Resolution of seed-level permutation tests
With `n` paired units the exact two-sided sign-flip p-value takes values in multiples of `2/2^n`; for
`n = 5` the minimum is `0.0625`. No result with five seeds can be "significant at 0.05" under this test,
independent of effect size. Effect sizes, CIs and the number of sign-consistent seeds carry the
evidence instead; more seeds (n >= 6 gives min p = 0.03125) are needed for a test-based claim.
