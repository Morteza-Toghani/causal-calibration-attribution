# 2. Introduction

Model-based reinforcement learning relies on learned world models that predict the next state from the current state and action. When such a model is trained on a fixed offline dataset, its predictions are used for planning or policy optimization in regions the data only partly covers, so the model must also report how uncertain it is [REF: Levine et al., offline RL tutorial; Kidambi et al., MOReL; Yu et al., MOPO; Janner et al., MBPO]. Probabilistic ensembles are a common choice for this purpose [REF: Lakshminarayanan et al., deep ensembles; Chua et al., PETS]. The usefulness of their uncertainty estimates depends on calibration: a nominal 90% predictive interval should contain the realized next state about 90% of the time [REF: Kuleshov et al., calibrated regression]. Miscalibrated uncertainty can lead a planner to trust predictions it should discount, or to discount predictions it could trust [REF: Malik et al., calibrated model-based RL].

Calibration is known to degrade under distribution shift [REF: Ovadia et al., can you trust your model's uncertainty]. Most studies, however, treat shift as one undifferentiated out-of-distribution signal. In an offline world-model setting, "shift" can mean at least three different things. The transition dynamics may differ from those that generated the data (dynamics shift). The inputs may be corrupted or measured differently (observation shift). The behavior that visits states and selects actions may change (policy shift). These mechanisms act on different parts of the prediction problem, and there is no reason to expect them to affect calibration equally, or even in the same direction. Detecting that calibration fails does not tell a practitioner which mechanism to guard against.

## The causal question

We ask:

> Which mechanism-specific distribution shift causally produces calibration failure in an uncertainty-aware offline world model, and by how much, under controlled intervention?

The question is one of attribution. We do not propose a new world model or a new detector. We hold the trained model fixed and ask how much each mechanism, applied alone, changes its calibration.

## Controlled counterfactual design

We trained 5-member Gaussian MLP ensembles on offline data from two MuJoCo locomotion environments, Hopper and Walker2d. Each mechanism was applied at evaluation time only, so no model was retrained per intervention. Dynamics shift scaled MuJoCo body masses; observation shift added Gaussian noise to the model input; policy shift scaled the actions applied at fixed probe states. Each intervention was applied at a low and a high severity. We compared each treated evaluation with an untreated one on the same trained model, the same instances, and shared random draws where noise was involved. The primary estimand was the paired average treatment effect (ATE) on an instance-level regression calibration error.

Two methodological issues arose that changed the conclusions, and we treat them as results. First, MuJoCo re-simulation of an identity intervention (mass scale 1.0, action scale 1.0) did not reproduce the dataset targets exactly, which produced a non-zero apparent effect for an intervention that should do nothing. Second, the trained ensembles were over-dispersed, so their intervals were wider than nominal at every level. We therefore report an analysis corrected for re-simulation bias and a second analysis that additionally recalibrates the baseline.

## Main finding

Once the baseline was recalibrated, observation shift was the dominant pathway of calibration failure in both environments. Its ATE was between +0.07749 and +0.26588 depending on environment and severity, while dynamics and policy effects did not exceed 0.01194 in magnitude. Before recalibration, the ranking was less clear, and the observation effect at high severity had opposite signs in Hopper and Walker2d. The naive hypothesis that every intervention increases calibration error was not supported: dynamics interventions reduced the metric in both environments, at both severities.

## Contributions

1. We formulate calibration failure under distribution shift as a mechanism-specific counterfactual attribution problem and implement it as evaluation-time interventions on a fixed trained ensemble, with matched instances and common random numbers.
2. We identify and quantify a re-simulation bias in MuJoCo-based interventions via a null (identity) control, and remove it by using a re-simulated paired baseline.
3. We show that the trained ensembles are over-dispersed at every nominal level, and that a single per-seed variance factor fitted on a held-out calibration split changes the ranking of mechanisms.
4. We report that, after recalibration, observation shift dominates dynamics and policy shift in both Hopper and Walker2d (n = 10 seeds each), with effect sizes and confidence intervals for all twelve environment-by-condition effects.
5. We provide a reproducible pipeline: the Hopper baseline reproduced the original pipeline's metrics to within 1.2 × 10⁻¹⁶ on five legacy seeds, and all numerical results derive from committed manifests.

## Scope of claims

The causal language in this paper refers to controlled computational interventions in a simulator on a fixed trained model. We make no claim about real-world causal identification. We tested two environments, one model class, and one severity pair per mechanism, and the limitations section details what this does not support.

## Organization

Section 3 reviews related work. Section 4 presents the method: model, interventions, metric, corrections, and statistical analysis. Section 5 gives the experimental setup. Section 6 reports results. Section 7 discusses interpretation and implications. Section 8 states limitations, and Section 9 concludes.
