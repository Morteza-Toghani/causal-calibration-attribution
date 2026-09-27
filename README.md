Counterfactual Causal Attribution of Calibration Failure in Uncertainty-Aware World Models

«MSc thesis research on understanding when and why uncertainty estimates become unreliable under mechanism-specific distribution shift.»

This repository contains the research code, experimental protocols, and analysis for a Master's thesis investigating whether different mechanisms of distribution shift produce systematically different calibration failures in an uncertainty-aware offline world model.

The study focuses on three mechanisms:

- Dynamics shift
- Observation shift
- Policy shift

The central goal is to design controlled comparisons that isolate the contribution of each mechanism to calibration failure.

---

Motivation

Uncertainty estimates are useful only when they remain meaningfully calibrated under the conditions in which an AI system is deployed.

Under distribution shift, calibration can degrade. However, distribution shift is not a single homogeneous phenomenon: changes in transition dynamics, observations, and policy-induced state distributions can affect the predictive pipeline through different mechanisms.

This project therefore studies mechanism-specific calibration failure rather than treating all distribution shift as one aggregate phenomenon.

The broader research direction is:

Uncertainty → Calibration → Calibration Failure → Mechanism-Specific Distribution Shift → Counterfactual Analysis → Statistical Validation → Recalibration → Reliable Decision-Making

---

Research Question

«Which mechanism-specific distribution shift is associated with calibration failure in an uncertainty-aware offline world model, and by how much under controlled intervention?»

The main comparison considers dynamics, observation, and policy interventions while keeping the evaluation setup as comparable as possible.

---

Mechanisms

1. Dynamics Shift

Controlled changes are applied to the transition dynamics at inference time while keeping the observation and policy components fixed.

2. Observation Shift

Controlled corruption or noise is applied to observations while keeping the underlying dynamics and policy fixed.

3. Policy Shift

The action-selection rule is changed on a fixed set of probe states while avoiding free rollout drift in the primary analysis.

This separation is important because the thesis is designed to distinguish the mechanism of shift from aggregate out-of-distribution severity.

---

Counterfactual Setup

The core experimental unit is a matched model–evaluation instance.

For each comparison:

- the trained world-model instance is kept fixed;
- evaluation states / probe states are matched;
- reusable sources of randomness are shared where appropriate;
- the intervention mechanism is changed;
- the resulting difference in calibration-related outcomes is measured.

The primary design uses Common Random Numbers (CRN) and paired comparisons so that variation unrelated to the intervention is reduced where possible.

The intended estimand for mechanism m is based on the paired outcome difference:

[
ATE_m = E[Y(m) - Y(0)]
]

where Y is the defined calibration-related outcome and 0 denotes the baseline condition.

The causal interpretation is limited to the controlled experimental design and its stated assumptions.

---

Method

World Model

A state-based probabilistic ensemble world model is used.

- Multilayer Perceptron (MLP)
- Ensemble size: K = 5
- Probabilistic prediction of the next state
- Gaussian predictive distributions
- Ensemble-based uncertainty estimation

For ensemble member k:

[
p_k(s_{t+1}\mid s_t,a_t)

\mathcal{N}(\mu_k,\sigma_k^2)
]

The predictive distribution is constructed from the ensemble members, separating within-model predictive uncertainty from ensemble disagreement.

Data

The project is designed around offline reinforcement-learning data with reproducible dataset metadata and manifests.

Raw datasets are not committed to Git. Dataset versions, environment information, and relevant metadata are recorded through configuration / manifest files.

Primary Prediction Horizon

The primary analysis uses 1-step transition prediction.

Multi-step evaluation is treated as an extension because rollout compounding and policy feedback can introduce additional sources of variation into attribution.

---

Calibration and Evaluation

The primary outcome is a regression-appropriate measure of calibration error.

Secondary metrics include:

- Predictive negative log-likelihood (NLL)
- Prediction-interval coverage
- Sharpness
- CRPS
- Predictive uncertainty
- Ensemble disagreement

The project distinguishes calibration from sharpness and does not treat a single metric as sufficient evidence of reliable uncertainty estimation.

---

Statistical Validation

The analysis is designed around matched observations and repeated model instances.

Planned statistical components include:

- Paired differences
- Multi-seed evaluation
- Bootstrap confidence intervals
- Paired permutation tests
- Effect sizes
- Holm correction for multiple comparisons

The analysis will report uncertainty around effect estimates rather than relying only on statistical significance.

---

Evidence Status

IN DESIGN

The experimental protocol, repository architecture, and core methodological design are being implemented and tested.

No final empirical claim is made here before the corresponding experiment, statistical analysis, and reproducibility checks are complete.

Evidence status will be updated as the project progresses:

"IN DESIGN → PROTOTYPE → PRELIMINARY → VALIDATED → REPRODUCIBLE"

---

Results

No final results are reported yet.

Once experiments are complete, this section will contain only results that can be traced to:

Research Question → Hypothesis → Intervention → Configuration → Experiment → Statistical Analysis → Result

Unsupported or placeholder numbers will not be included.

---

Limitations

The main limitations currently considered include:

- controlled simulation does not establish causal effects in real-world deployment;
- causal interpretation depends on the intervention design and its assumptions;
- policy intervention in the primary analysis is evaluated on fixed probe states rather than unrestricted rollout;
- calibration behaviour may depend on model architecture, environment, dataset, and random seed;
- multi-step uncertainty propagation is outside the primary thesis-level outcome.

These limitations will be updated as the experimental evidence develops.

---

Reproducibility

The repository is designed so that experiments can be reconstructed from explicit configurations rather than undocumented manual steps.

Each experiment should record:

- environment
- dataset version / manifest
- model configuration
- random seed
- intervention mechanism
- intervention severity
- output location

Large datasets remain outside Git; only manifests and reproducibility metadata belong in the repository.

---

Repository Structure

causal-calibration-world-models/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── model/
│   ├── shift/
│   └── experiment/
│
├── data/
│   └── manifests/
│
├── src/
│   ├── envs/
│   ├── models/
│   ├── shifts/
│   ├── metrics/
│   ├── causal/
│   └── stats/
│
├── experiments/
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── notebooks/
├── docs/
└── tests/

The "causal/" layer is explicit because causal attribution is a core methodological component of the thesis rather than a secondary analysis hidden inside notebooks.

The "docs/" directory will document the counterfactual design, causal assumptions, methodology, limitations, and related work.

---

Research Scope

This repository corresponds primarily to the MSc thesis / Paper 1 diagnosis stage of a broader research program:

Thesis → Paper 1 → Paper 2 → PhD

The thesis establishes the mechanism-specific calibration-failure and attribution framework.

Later work may extend the diagnosis toward causally-informed recalibration and safety-constrained decision-making, but those components are not treated as completed contributions in this repository.

---

Author
Morteza Toghani
MSc Artificial Intelligence & Robotics
