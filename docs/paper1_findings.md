# Paper 1 — Key scientific findings

## Core result

After **post-hoc recalibration** of the baseline ensemble on the
calibration split (variance scaling with `alpha ≈ 0.24` across all 5
seeds), the effect of mechanism-specific shift on calibration is
**dominated by the observation pathway**:

| Condition | ATE (recalibrated) | 95% t-CI | Excludes 0? |
|---|---|---|---|
| observation_low | **+0.199** | [+0.194, +0.203] | YES |
| observation_high | **+0.264** | [+0.261, +0.267] | YES |
| dynamics_high | -0.0038 | [-0.0059, -0.0017] | YES |
| dynamics_low | -0.0012 | [-0.0016, -0.0008] | YES |
| policy_high | +0.0027 | [+0.0018, +0.0037] | YES |
| policy_low | +0.0008 | [+0.0004, +0.0012] | YES |

The observation effects are **two orders of magnitude larger** than the
dynamics or policy effects. In a properly calibrated baseline, the
observation shift is the only mechanism with a practically relevant
effect on calibration.

## Two additional scientific findings

### 1. Baseline over-dispersion (α ≈ 0.24)

Fitted on the calibration split (nominal level 0.90), the per-seed
variance scaling factor is `alpha ∈ [0.222, 0.260]`. This means the
baseline predictive variance is **≈4x too large**.

Signed coverage of the baseline at nominal level 0.50 is `+0.35`, i.e.,
coverage is 85% when nominal is 50%. The model over-covers strongly.

### 2. Re-simulation bias in dynamics and policy interventions

An identity intervention (`mass_scale=1.0` for dynamics, `action_scale=1.0`
for policy) does **not** reproduce the dataset target bit-for-bit. The
per-condition ATE bias is `≈ -0.0032` across all 5 seeds.

Root cause: MuJoCo `set_state` + `step` does not reset the internal
`qacc_warmstart`, and the x=0 reconstruction of state from observation
introduces a systematic shift.

**Correction:** use `y_resim(scale=1.0)` as the paired baseline target
for D and P interventions, not the dataset `y`. After correction, the
dynamics and policy ATEs change substantially:

| Condition | Original | Corrected | Recalibrated |
|---|---|---|---|
| dynamics_high | -0.0248 | -0.0216 | -0.0038 |
| dynamics_low  | -0.0059 | -0.0028 | -0.0012 |
| policy_high   | +0.0034 | +0.0066 | +0.0027 |
| policy_low    | +0.0002 | +0.0035 | +0.0008 |

The **original reported values understate the policy effect and
overstate the dynamics effect**. Under the recalibrated baseline,
both are small.

## Reproduction

All results in this document are reproducible from the repository:

```bash
python scripts/reproduce_baseline_all_seeds.py         # baseline (5 seeds)
python scripts/reproduce_interventions_all_seeds.py    # interventions (5 seeds, uncorrected)
python scripts/null_control.py                         # reveals re-simulation bias
python scripts/analyze_signed_coverage.py              # direction-aware coverage
python scripts/analyze_interventions_corrected.py      # corrected ATEs
python scripts/analyze_t_ci.py                         # t-CI on corrected
python scripts/recalibrate_and_rerun.py                # recalibration
python scripts/analyze_t_ci_recalibrated.py            # t-CI on recalibrated

What this means for the thesis
The original research question — "which mechanism-specific shift causally
produces calibration failure?" — has a clean, quantifiable answer:
the observation pathway.

Under a calibrated baseline:

Observation shift produces calibration failure with ATE ~ 0.20-0.26.

Dynamics shift produces a small positive/negative effect (~10⁻³).

Policy shift produces a small positive effect (~10⁻³).

The apparent heterogeneity in the earlier analysis was largely an
artifact of the over-dispersed baseline. Once calibrated, a single
pathway dominates.