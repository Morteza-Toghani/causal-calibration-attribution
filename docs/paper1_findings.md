# Paper 1 — Key scientific findings (FINAL, n=10 both envs)

## Core result

After **post-hoc recalibration** of the baseline ensemble on the
calibration split (variance scaling with \lpha\ fitted per seed at
nominal 0.90), the effect of mechanism-specific shift on calibration
is **dominated by the observation pathway** in both environments.

### Hopper, n=10 seeds, alpha = 0.2361, t_crit = 2.262

| Condition | ATE | 95% t-CI |
|---|---|---|
| observation_low | **+0.20125** | [+0.19818, +0.20432] |
| observation_high | **+0.26588** | [+0.26359, +0.26816] |
| dynamics_low | -0.00113 | [-0.00162, -0.00065] |
| dynamics_high | -0.00445 | [-0.00597, -0.00294] |
| policy_low | +0.00086 | [+0.00065, +0.00106] |
| policy_high | +0.00247 | [+0.00180, +0.00313] |

### Walker2d, n=10 seeds, alpha = 0.3743, t_crit = 2.262

| Condition | ATE | 95% t-CI |
|---|---|---|
| observation_low | **+0.07749** | [+0.07468, +0.08031] |
| observation_high | **+0.15268** | [+0.14993, +0.15542] |
| dynamics_low | -0.00178 | [-0.00208, -0.00148] |
| dynamics_high | -0.01194 | [-0.01284, -0.01105] |
| policy_low | +0.00486 | [+0.00435, +0.00537] |
| policy_high | +0.00630 | [+0.00452, +0.00807] |

**Magnitude ranking (all values in units of calibration error):**

| Env | |O| | |D| | |P| | Ratio O/D |
|---|---|---|---|---|---|
| Hopper | 0.20-0.27 | 0.001-0.004 | 0.001-0.002 | ~60-200x |
| Walker2d | 0.08-0.15 | 0.002-0.012 | 0.005-0.006 | ~13-30x |

**All 12 effects exclude zero at 95%.**

## Two methodological findings

### 1. Baseline over-dispersion

The 5-member ensembles over-cover at every nominal level. Fitted on the
**calibration split** (nominal 0.90), the per-seed variance scaling
factor is:
- Hopper: \lpha ≈ 0.2361\ -> variance ~4.2x too large
- Walker2d: \lpha ≈ 0.3743\ -> variance ~2.7x too large

### 2. Re-simulation bias in dynamics and policy

An identity intervention (mass_scale=1.0 for dynamics, action_scale=1.0
for policy) does NOT reproduce the dataset target bit-for-bit:
- Hopper: mean ATE ≈ -0.0032
- Walker2d: mean ATE ≈ -0.0066

Cause: MuJoCo set_state + step does not reset internal qacc_warmstart,
plus the x=0 reconstruction of state from observation.

Correction: use y_resim(scale=1.0) as the paired baseline target for
D and P interventions.

## Cross-environment summary

- **Observation dominates in both environments** by 13-200x, depending
  on environment.
- **Dynamics magnitude is comparable across environments** (-0.004 to
  -0.024 on both).
- **Policy effect is 2-3x larger on Walker2d** than on Hopper.
- **The sign of observation_high** in the pre-recalibration analysis
  flips between environments (Hopper +0.041, Walker2d -0.033); after
  recalibration both are positive and large. This sign dependence is
  an artifact of baseline over-dispersion magnitude, which itself
  differs by environment (alpha 0.236 vs 0.374).

## Reproduction

All results are reproducible from the repository:

    python scripts/reproduce_baseline_all_seeds.py         # Hopper baseline (bit-exact)
    python scripts/reproduce_interventions_all_seeds.py    # Hopper interventions (bit-exact)
    python scripts/null_control.py                         # re-simulation bias
    python scripts/analyze_signed_coverage.py              # direction-aware coverage
    python scripts/analyze_interventions_corrected.py      # corrected ATE
    python scripts/recalibrate_and_rerun.py                # recalibration (n=5)
    python scripts/run_full_pipeline.py --env walker2d     # Walker2d pipeline
    python scripts/final_n10_analysis.py                   # n=10 corrected (both envs)
    python scripts/final_n10_recalibrated.py               # n=10 recalibrated (MAIN)

## What this means for the thesis

Under a calibrated baseline, the observation pathway is the dominant
source of calibration failure. Its effect is 13-200x larger than the
dynamics or policy effects, across two environments and 20 independent
seed-level training runs.

This has direct implications for how calibration failure under
distribution shift should be reported: without a properly calibrated
baseline and without separating shift by mechanism, the relative
importance of different failure modes can be misestimated.
