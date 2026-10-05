# Limitations (what is not yet established)

1. **Single environment, five seeds.** Hopper only; Walker2d replication not started.
2. **Original experiment code is not in this repository.** Raw results are included with checksums, and
   everything downstream is regenerated here, but model training and the dynamics/policy interventions
   cannot yet be rerun from this repo.
3. **Pair-level ATE differs from the aggregate metric difference** (`results/tables/ate_vs_aggregate_difference.md`,
   gaps up to 0.010 for observation low). The pair unit used by the original run is not documented here.
4. **Severity is not matched across mechanisms**, evaluation-pair counts differ (5,000 / 25,000 / 10,000),
   and dynamics/policy intensities are undocumented here. No mechanism ranking is supported.
5. **Unsigned primary metric** hides the direction of miscalibration (`theoretical_analysis.md` section 3).
6. **Baseline over-covers strongly** (pooled coverage 0.85 at nominal 0.50), so effects are measured
   relative to an already miscalibrated reference, and the baseline configuration is not frozen.
7. **Seed-level tests are resolution-limited** (min p = 0.0625 for n = 5); Holm-adjusted p-values are >= 0.375.
8. **Test-set reuse**: the test split was used for pilot diagnostics (`methodology.md`).
9. **Intervention validity untested**: mechanism-specificity and cross-effects not yet audited.
10. **Gaussian-MLP world model, one dataset family, one-step horizon**: external validity is unknown.
11. **Observation-shift prototype** assumes a noise reference scale that has not been verified against the original run.
