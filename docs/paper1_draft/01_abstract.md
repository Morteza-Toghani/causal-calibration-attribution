# Abstract (draft)

Uncertainty-aware world models are increasingly used for decision-making
under uncertainty, but their calibration can break down under
distribution shift. Existing work typically treats distribution shift
as a single, undifferentiated source of failure, reporting calibration
degradation without attributing it to a specific mechanism. We ask a
different question: **which mechanism-specific shift causally produces
calibration failure, and by how much, under controlled intervention?**

We study three mechanisms — dynamics shift, observation shift, and
policy shift — in a 5-member probabilistic ensemble world model trained
on the Minari `mujoco/hopper/medium-v0` and `mujoco/walker2d/medium-v0`
offline datasets. We apply each intervention at evaluation time, on
fixed probe states, with paired random draws (CRN) and report paired
average treatment effects (ATE) on regression calibration error over
5 training seeds.

Three findings emerge. First, the baseline ensemble over-covers at
every nominal level; a single per-seed variance scaling factor
`alpha` fitted on a calibration split (never used during training)
brings coverage to nominal (`alpha ≈ 0.24` on Hopper, `≈ 0.37` on
Walker2d). Second, an identity intervention reveals a systematic
re-simulation bias in the dynamics and policy pipelines
(`≈ -0.003` on Hopper, `≈ -0.007` on Walker2d), which we correct by
using re-simulated targets as the paired baseline for both conditions.
Third, after correction and recalibration, the observation pathway
dominates: its effect on calibration error is two orders of magnitude
larger than the dynamics or policy effects on Hopper, and three to
five times larger on Walker2d.

Our results show that mechanism attribution of calibration failure
requires (a) a properly calibrated baseline, (b) accounting for
re-simulation bias, and (c) direction-aware evaluation. We also
observe that the sign of the observation effect depends on the
magnitude of baseline over-dispersion, which may explain why earlier
studies have reported inconsistent results.