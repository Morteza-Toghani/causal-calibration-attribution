# Negative results (real outcomes only)

Items N1-N4 are taken from the author's pilot report (Hopper, single seed, test-set diagnostics, not
regenerable from this repository). N5 is regenerated from `results/raw` by `scripts/analyze_results.py`.

### N1 - "The ensemble is merely under-trained" (training length)
Hypothesis: Longer training (50 vs 20 epochs) improves calibration.
Experiment: Config A (min_log_var -10, 20 epochs) vs config B (min_log_var -10, 50 epochs).
Result: NLL improved (-3.64 to -4.66) but calibration error worsened (0.264 to 0.278); coverage at nominal 0.50 rose from 0.933 to 0.959.
Statistical Evidence: one seed, descriptive only.
Interpretation: Better likelihood does not imply better interval calibration here.
Action: Do not choose configurations by NLL alone; baseline remains unfrozen.

### N2 - "The lower variance clamp explains the over-coverage"
Hypothesis: The variance floor causes the over-coverage.
Experiment: Lower the floor (min_log_var -10 to -20, config C).
Result: Calibration error improved (0.264 to 0.235) but coverage at nominal 0.50 stayed at 0.877; with the floor inactive (no lower saturation) over-coverage remained.
Statistical Evidence: one seed, descriptive only.
Interpretation: The floor contributes but does not explain the failure.
Action: Keep investigating the variance head; do not describe the floor as the cause.

### N3 - "A normalisation bug causes the miscalibration"
Result: Not confirmed as the main cause (pilot audit).
Action: Documented and closed unless new evidence appears.

### N4 - "The upper variance clamp saturates and distorts the variance head"
Result: Upper-clamp saturation was very small (about 0.09% of raw log-variances in one diagnostic).
Action: Not pursued as an explanation.

### N5 - "Every intervention worsens calibration error" (H1 in its simplest form)
Hypothesis: ATE on calibration error is positive for each mechanism and intensity.
Experiment: Hopper, 5 seeds, six conditions (`results/tables/mechanism_attribution_summary.md`).
Result: Not supported. ATE is negative in all 5 seeds for dynamics low, dynamics high and observation low; positive in all 5 seeds only for observation high; policy effects are near zero.
Statistical Evidence: n = 5 seeds; sign-flip p >= 0.0625, Holm-adjusted p >= 0.375; bootstrap CIs of the mean ATE exclude zero for the dynamics and observation conditions and policy high.
Interpretation: The baseline already over-covers, so shifts that reduce coverage lower an unsigned error. For observation low the sign of the bias reverses (over- to under-coverage), which the unsigned metric hides.
Action: Report signed bias alongside calibration error; reformulate H1 in terms of coverage direction.
