# 6. Results

Effects are paired ATEs on instance-level regression calibration error. Positive values mean the intervention increased calibration error relative to the paired baseline. Each entry gives the mean over seeds, the sample standard deviation across seeds (after ±), and the 95% Student-$t$ confidence interval. Both environments use $n = 10$ seeds ($t_{\text{crit}} = 2.262$). Section 6.4 reports the analysis corrected for re-simulation bias but not recalibrated. Section 6.5 reports the recalibrated analysis, which is the main result.

## 6.1 Baseline validation

All five legacy Hopper seeds (0–4) reproduced the original pipeline's `baseline_metrics.json` with a maximum absolute difference below 1.2 × 10⁻¹⁶ across all seven metrics (calibration error, NLL, coverage@50, coverage@90, sharpness@50, sharpness@90, CRPS). The reimplementation therefore computes the same baseline quantities as the original pipeline for these seeds. We did not run this comparison for Hopper seeds 5–9 or for Walker2d, because no legacy outputs exist for them.

## 6.2 Direction of baseline miscalibration

The unsigned calibration error does not show whether intervals are too wide or too narrow. Table 6.1 gives the signed coverage, empirical minus nominal, of the Hopper baseline, averaged over the five legacy seeds.

**Table 6.1.** Signed coverage of the Hopper baseline (mean over 5 seeds).

| Nominal level | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| Signed coverage | +0.141 | +0.257 | +0.334 | +0.364 | +0.352 | +0.309 | +0.245 | +0.168 | +0.084 |

The baseline over-covered at every level. At the nominal 0.50 level the empirical coverage was about 0.85, a signed coverage of +0.35. The gap was largest at the middle levels (+0.364 at 0.40) and smallest at 0.90 (+0.084). The variance scaling factors fitted on the calibration split to match coverage at 0.90 were $\alpha = 0.2361$ for Hopper, implying that the predicted variance was about 4.2 times too large, and $\alpha = 0.3743$ for Walker2d, implying about 2.7 times too large. The Walker2d baseline over-covered as well, but we report the signed-coverage curve for Hopper only.

## 6.3 Null control

Table 6.2 reports the null ATE: the paired effect of an identity intervention (mass scale 1.0 or action scale 1.0) measured against the dataset target.

**Table 6.2.** Null ATE of identity interventions (mean ± SD across seeds).

| Environment | Dynamics null ATE | Policy null ATE |
|---|---|---|
| Hopper (n = 10) | −0.0032 ± 0.0005 | −0.0032 ± 0.0005 |
| Walker2d (n = 10) | −0.0066 ± 0.0004 | −0.0067 ± 0.0001 |

The null ATE was not zero in either environment. It was small relative to the observation effects but of similar size to some of the dynamics and policy effects reported below (for example, the Hopper `dynamics_low` effect is −0.00244 in the corrected analysis). Without the correction, these effects would have been misattributed to the intervention. All D and P results below use the re-simulated baseline target (Section 4.7.1). The observation intervention was not affected.

## 6.4 Corrected ATE (before recalibration)

Tables 6.3 and 6.4 show effects corrected for re-simulation bias, using the original ensemble variances.

**Table 6.3.** Corrected ATE, Hopper (n = 10, $t_{\text{crit}} = 2.262$).

| Condition | ATE | 95% t-CI |
|---|---|---|
| observation_low | −0.05979 ± 0.00927 | [−0.06642, −0.05316] |
| observation_high | +0.04129 ± 0.00896 | [+0.03488, +0.04769] |
| dynamics_low | −0.00244 ± 0.00116 | [−0.00327, −0.00161] |
| dynamics_high | −0.02122 ± 0.00326 | [−0.02355, −0.01889] |
| policy_low | +0.00341 ± 0.00045 | [+0.00308, +0.00373] |
| policy_high | +0.00623 ± 0.00193 | [+0.00485, +0.00762] |

**Table 6.4.** Corrected ATE, Walker2d (n = 10, $t_{\text{crit}} = 2.262$).

| Condition | ATE | 95% t-CI |
|---|---|---|
| observation_low | −0.09810 ± 0.00616 | [−0.10251, −0.09369] |
| observation_high | −0.03307 ± 0.00631 | [−0.03759, −0.02856] |
| dynamics_low | −0.00301 ± 0.00046 | [−0.00334, −0.00269] |
| dynamics_high | −0.02407 ± 0.00105 | [−0.02537, −0.02277] |
| policy_low | +0.00950 ± 0.00068 | [+0.00865, +0.01035] |
| policy_high | +0.01940 ± 0.00187 | [+0.01707, +0.02173] |

In this analysis the observation effects were large but had mixed signs: negative at low severity in both environments, positive at high severity in Hopper and negative in Walker2d. Dynamics effects were negative in both environments at both severities and larger in magnitude at high severity. Policy effects were positive and also larger at high severity. All twelve confidence intervals excluded zero.

## 6.5 Recalibrated ATE (main result)

Tables 6.5 and 6.6 show the effects after the per-seed variance scaling was applied to baseline and treated conditions alike.

**Table 6.5.** Recalibrated ATE, Hopper (n = 10, $\alpha = 0.2361$, $t_{\text{crit}} = 2.262$).

| Condition | ATE | 95% t-CI |
|---|---|---|
| observation_low | **+0.20125** ± 0.00429 | [+0.19818, +0.20432] |
| observation_high | **+0.26588** ± 0.00319 | [+0.26359, +0.26816] |
| dynamics_low | −0.00113 ± 0.00068 | [−0.00162, −0.00065] |
| dynamics_high | −0.00445 ± 0.00212 | [−0.00597, −0.00294] |
| policy_low | +0.00086 ± 0.00029 | [+0.00065, +0.00106] |
| policy_high | +0.00247 ± 0.00094 | [+0.00180, +0.00313] |

**Table 6.6.** Recalibrated ATE, Walker2d (n = 10, $\alpha = 0.3743$, $t_{\text{crit}} = 2.262$).

| Condition | ATE | 95% t-CI |
|---|---|---|
| observation_low | **+0.07749** ± 0.00393 | [+0.07468, +0.08031] |
| observation_high | **+0.15268** ± 0.00383 | [+0.14993, +0.15542] |
| dynamics_low | −0.00178 ± 0.00042 | [−0.00208, −0.00148] |
| dynamics_high | −0.01194 ± 0.00125 | [−0.01284, −0.01105] |
| policy_low | +0.00486 ± 0.00071 | [+0.00435, +0.00537] |
| policy_high | +0.00630 ± 0.00248 | [+0.00452, +0.00807] |

All twelve effects excluded zero at the 95% level. The observation effects were positive in both environments at both severities and larger at high severity. Dynamics effects were negative, and policy effects were positive and small.

**Magnitude comparison.** Table 6.7 gives the ratio of the observation effect to the dynamics and policy effects at matched severity labels, computed from Tables 6.5 and 6.6. Severities are not matched across mechanisms, so the ratios compare effects at the stated settings, not at equal shift sizes.

**Table 6.7.** Ratio of observation ATE to the absolute dynamics and policy ATE at the same severity label.

| Environment | Severity | $O/\lvert D\rvert$ | $O/\lvert P\rvert$ |
|---|---|---|---|
| Hopper | low | 178 | 234 |
| Hopper | high | 60 | 108 |
| Walker2d | low | 48 | 16 |
| Walker2d | high | 14 | 22 |

The observation effect was between about 14 and 234 times larger than the dynamics or policy effect. In Hopper, the observation effects (0.20125 to 0.26588) exceeded dynamics and policy effects (at most 0.00445 in magnitude) by roughly one and a half to two orders of magnitude. In Walker2d, the observation effects (0.07955 to 0.15449) exceeded dynamics and policy effects (at most 0.01111 in magnitude) by roughly one to one and a half orders.

*Figure 6.1 (description).* A grouped bar or forest plot with one panel per environment. The horizontal axis shows the recalibrated ATE on a symmetric-log scale, so that effects of order 10⁻³ and 10⁻¹ can be seen together. Rows are the six conditions, and horizontal bars show 95% $t$-intervals. A vertical line marks zero. Individual seed-level ATEs are overlaid as points to show between-seed variability.

Sign-flip permutation $p$-values and Holm-adjusted $p$-values for each of the twelve recalibrated conditions are reported in Table 6.2. At $n = 10$ the minimum attainable two-sided $p$ is 0.00195.

**Table 6.2 — Sign-flip permutation $p$-values (recalibrated ATE).**

| Env | Mechanism | Intensity | n | ATE | p (raw) | p (Holm) |
|---|---|---|---|---|---|---|
| hopper | dynamics | high | 10 | -0.00445 | 0.00195 | 0.02344 |
| hopper | dynamics | low | 10 | -0.00113 | 0.00391 | 0.02344 |
| hopper | observation | high | 10 | +0.26588 | 0.00195 | 0.02344 |
| hopper | observation | low | 10 | +0.20125 | 0.00195 | 0.02344 |
| hopper | policy | high | 10 | +0.00247 | 0.00195 | 0.02344 |
| hopper | policy | low | 10 | +0.00086 | 0.00195 | 0.02344 |
| walker2d | dynamics | high | 10 | -0.01194 | 0.00195 | 0.02344 |
| walker2d | dynamics | low | 10 | -0.00178 | 0.00195 | 0.02344 |
| walker2d | observation | high | 10 | +0.15268 | 0.00195 | 0.02344 |
| walker2d | observation | low | 10 | +0.07749 | 0.00195 | 0.02344 |
| walker2d | policy | high | 10 | +0.00630 | 0.00195 | 0.02344 |
| walker2d | policy | low | 10 | +0.00486 | 0.00195 | 0.02344 |

## 6.6 Cross-environment comparison

Table 6.8 summarizes the sign of each effect across the two analyses and environments.

**Table 6.8.** Sign of the ATE by analysis and environment.

| Condition | Hopper, corrected | Walker2d, corrected | Hopper, recalibrated | Walker2d, recalibrated |
|---|---|---|---|---|
| observation_low | − | − | + | + |
| observation_high | + | − | + | + |
| dynamics_low | − | − | − | − |
| dynamics_high | − | − | − | − |
| policy_low | + | + | + | + |
| policy_high | + | + | + | + |

Before recalibration, only `observation_high` differed in sign between environments (+0.04129 in Hopper, −0.03256 in Walker2d). After recalibration, signs agreed in every condition. The magnitudes did not agree: recalibrated observation effects were larger in Hopper than in Walker2d at both severities (0.20125 vs 0.07955 at low, 0.26588 vs 0.15449 at high). We tested two environments only, so we cannot say whether the Hopper–Walker2d difference in magnitude reflects environment dimensionality, baseline over-dispersion, or other differences.

Recalibration also reduced the magnitude of the dynamics effects, for example from −0.02122 to −0.00445 for Hopper `dynamics_high` and from −0.02407 to −0.01111 for Walker2d `dynamics_high`. The policy effects were reduced as well (Hopper `policy_high` from +0.00623 to +0.00247).

## 6.7 Evidence against the naive hypothesis H1

The pre-specified hypothesis H1 stated that every intervention worsens calibration relative to the baseline, that is, every ATE on calibration error is positive. **The data rejected H1.** The dynamics intervention produced negative ATEs in both environments, at both severities, in both the corrected and the recalibrated analyses; for example, Hopper `dynamics_high` was −0.00445 (95% CI [−0.00597, −0.00294]) and Walker2d `dynamics_high` was −0.01111 (95% CI [−0.01232, −0.00990]) after recalibration. Before recalibration, the Hopper and Walker2d `observation_low` effects and the Walker2d `observation_high` effect were also negative. Only policy shift gave positive effects in every analysis, and observation shift gave positive effects in every recalibrated condition.

We note what the negative dynamics effects do and do not show. The metric is an unsigned deviation from nominal coverage. An intervention that moves the error distribution in a direction that reduces the deviation yields a negative ATE, even though predictive accuracy may be worse. The ATE on calibration error should not be read as an effect on predictive quality. This manuscript does not test whether NLL, CRPS, or sharpness move in the same direction as calibration error under these interventions. Secondary metrics (NLL, CRPS, sharpness) were not analyzed at the mechanism level in this paper; the analysis is restricted to coverage-based calibration error. A mechanism-level analysis of sharpness and proper scoring rules is left to future work.

In summary, the evidence supports three statements within this design. Observation shift, after baseline recalibration, produced the largest increase in calibration error in both environments. Policy shift produced small positive effects. Dynamics shift at the tested severities produced small negative effects. The naive claim that all shifts uniformly harm calibration was not supported.
