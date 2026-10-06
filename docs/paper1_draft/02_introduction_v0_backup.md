# 1. Introduction (draft)

Uncertainty-aware world models — probabilistic predictors of the
next state of a dynamical system — are increasingly used as components
of decision-making pipelines in reinforcement learning, robotics, and
model-based control [refs]. For these models to be useful, their
uncertainty estimates must be *calibrated*: the predicted distribution
of the next state should match the empirical distribution of realized
next states.

Under distribution shift, this calibration can break down. Prior work
has documented this breakdown empirically and proposed various
recalibration methods [refs]. However, existing studies typically
treat distribution shift as a single, undifferentiated signal — an
"OOD score" or an "out-of-distribution" flag — and report calibration
degradation without asking *which specific change in the data-generating
process* is responsible for the failure.

This matters because different changes act on different parts of the
predictive pipeline. A change in the transition dynamics alters the
ground-truth transition function while leaving the observation process
and the policy unchanged. A change in the observation process corrupts
the inputs to the model while leaving the underlying physics intact.
A change in the policy alters which states and actions are being
evaluated, which in turn shifts the state distribution that the model
sees. Treating these mechanisms as interchangeable conflates physically
and statistically distinct phenomena.

In this paper, we ask a causal question: **which mechanism-specific
distribution shift causally produces calibration failure in an
uncertainty-aware offline world model, and by how much, under
controlled intervention?**

To answer this question we design a controlled counterfactual
evaluation protocol:

1. We train a 5-member probabilistic Gaussian MLP ensemble on offline
   RL data (`mujoco/hopper/medium-v0`, `mujoco/walker2d/medium-v0`),
   with fixed hyperparameters and 5 independent training seeds.

2. For each seed, we apply each mechanism-specific intervention at
   *evaluation time only*, keeping the trained model fixed. This
   isolates the effect of the intervention from training-time
   variability.

3. We use **fixed probe states** for the policy intervention, so that
   the probe set is identical across baseline and treatment.

4. We use **paired random draws** (Common Random Numbers) between
   baseline and treatment conditions, so that the paired difference
   cancels nuisance randomness.

5. We compute a mechanism-specific **paired average treatment effect**
   on a direction-aware, per-instance regression calibration error,
   and report 95% confidence intervals using the Student-t distribution
   with `df = 4`.

Our main finding is that, once the baseline is properly calibrated and
the re-simulation bias in the dynamics and policy pipelines is removed,
the observation pathway dominates: its effect on calibration error is
two orders of magnitude larger than the dynamics or policy effects on
Hopper, and three to five times larger on Walker2d.

We additionally document two methodological pitfalls that, if not
handled, can produce misleading results: (a) baseline over-dispersion
can cause the sign of the observation effect to flip, and (b) a
re-simulation bias in the dynamics and policy pipelines leaks into the
paired ATE unless the paired baseline is also re-simulated.

**Contributions.**

1. A modular, reproducible implementation of a controlled counterfactual
   evaluation protocol for calibration failure in uncertainty-aware
   world models.

2. A direction-aware, per-instance regression calibration error that
   supports paired ATE analysis.

3. An empirical identification and correction of a re-simulation bias
   in the dynamics and policy intervention pipelines.

4. A controlled comparison of three mechanism-specific shifts across
   two environments, showing that the observation pathway dominates.

5. A cross-environment observation about the sign of the observation
   effect depending on baseline over-dispersion.

**Paper organization.** Section 2 reviews related work. Section 3
describes our method. Section 4 describes the experimental setup.
Section 5 reports results. Section 6 discusses limitations and
implications. Section 7 concludes.