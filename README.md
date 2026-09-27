# Counterfactual Causal Attribution of Calibration Failure in Uncertainty-Aware World Models

> **MSc Thesis Research · Artificial Intelligence & Robotics**

**Investigating when and why uncertainty estimates become unreliable under mechanism-specific distribution shift, with a focus on calibration, controlled interventions, and causal / counterfactual analysis.**

---

## Overview

Uncertainty estimates are useful only when they remain meaningfully calibrated under the conditions in which an AI system is evaluated or deployed.

Under **distribution shift**, calibration can degrade. However, distribution shift is not a single homogeneous phenomenon: changes in **dynamics**, **observations**, and **policy-induced state distributions** can affect different parts of the predictive pipeline.

This project therefore studies **mechanism-specific calibration failure** rather than treating all distribution shift as one aggregate phenomenon.

The central research direction is:

**Uncertainty → Calibration → Calibration Failure → Mechanism-Specific Distribution Shift → Counterfactual Analysis → Statistical Validation → Recalibration → Reliable Decision-Making**

---

## Research Question

> **Which mechanism-specific distribution shift is associated with calibration failure in an uncertainty-aware offline world model, and by how much under controlled intervention?**

The primary comparison considers three mechanisms:

* **Dynamics shift**
* **Observation shift**
* **Policy shift**

The goal is to determine whether controlled changes in these mechanisms produce systematically different calibration outcomes.

---

## Mechanism-Specific Shift

### Dynamics Shift

A controlled change is applied to the **transition dynamics** during evaluation while keeping the observation and policy components fixed.

### Observation Shift

A controlled **corruption / noise transformation** is applied to observations while keeping the underlying dynamics and policy fixed.

### Policy Shift

The **action-selection rule** is changed on a fixed set of probe states while avoiding unrestricted rollout drift in the primary analysis.

This separation is central to the study because the objective is to investigate the contribution of the **shift mechanism itself**, rather than only comparing aggregate levels of out-of-distribution severity.

---

## Counterfactual Setup

The core experimental unit is a **matched model–evaluation instance**.

For each comparison:

1. The trained world-model instance is kept fixed.
2. The evaluation setup is matched as closely as possible.
3. Reusable sources of randomness are shared where appropriate.
4. One shift mechanism is intervened on.
5. The resulting change in calibration-related outcomes is measured relative to baseline.

The primary design uses **Common Random Numbers (CRN)** and **paired comparisons** to reduce variation unrelated to the intervention.

The intended mechanism-level estimand is:

$$
ATE_m = E[Y(m) - Y(0)]
$$

where:

* \(m\) denotes the intervention mechanism,
* \(Y\) is the defined calibration-related outcome,
* \(0\) denotes the baseline condition.

The causal interpretation is restricted to the controlled experimental design and its documented assumptions.

---

## Method

### Uncertainty-Aware World Model

The project uses a **state-based probabilistic ensemble world model**.

**Model design**

* Multilayer Perceptron (**MLP**)
* Ensemble size: **K = 5**
* Probabilistic next-state prediction
* Gaussian predictive distributions
* Ensemble-based uncertainty estimation

For ensemble member \(k\):

$$
p_k(s_{t+1}\mid s_t,a_t)
=
\mathcal{N}(\mu_k,\sigma_k^2)
$$

The ensemble combines predictive behaviour across members to represent uncertainty while distinguishing within-model predictive variability from ensemble disagreement.

### Data

The study is designed around **offline reinforcement-learning data** with reproducible dataset metadata and manifests.

Raw datasets are **not committed to Git**.

Dataset versions, environment information, and relevant metadata are intended to be recorded through configuration and manifest files.

### Primary Prediction Horizon

The primary analysis uses **1-step transition prediction**.

Multi-step evaluation is treated as an extension because rollout compounding and policy feedback can introduce additional sources of variation into mechanism attribution.

---

## Calibration & Evaluation

The primary outcome is a **regression-appropriate calibration measure**.

Secondary evaluation includes:

* **Negative Log-Likelihood (NLL)**
* **Prediction-interval coverage**
* **Sharpness**
* **CRPS**
* **Predictive variance**
* **Ensemble disagreement**
* **Calibration curves**
* **Error–uncertainty relationship**

Calibration and sharpness are treated as complementary properties of probabilistic predictions rather than interchangeable measures.

---

## Statistical Validation

The experimental design is based on matched observations and repeated model instances.

Planned statistical components include:

* **Paired differences**
* **Multi-seed evaluation**
* **Bootstrap confidence intervals**
* **Paired permutation tests**
* **Effect sizes**
* **Holm correction for multiple comparisons**

The analysis will emphasize **effect estimates and uncertainty intervals**, rather than relying only on statistical significance.

---

## Evidence Status

> 🚧 **IN DESIGN / PILOT PHASE**

The research protocol, experimental design, and repository architecture are being implemented and tested.

At this stage, the repository does **not** claim final empirical findings.

Evidence status will evolve according to the project's actual maturity:

`IN DESIGN → PROTOTYPE → PRELIMINARY → VALIDATED → REPRODUCIBLE`

No status will be upgraded without the corresponding experimental and statistical evidence.

---

## Results

### No final results reported yet.

Results will be added only when they can be traced through the full research chain:

**Research Question → Hypothesis → Intervention → Configuration → Experiment → Statistical Analysis → Result**

No placeholder numbers, unsupported claims, or illustrative results will be presented as empirical findings.

---

## Limitations

The main limitations currently considered include:

* Controlled simulation does not establish causal effects in real-world deployment.
* Causal interpretation depends on the intervention design and its assumptions.
* The primary policy intervention uses fixed probe states rather than unrestricted rollout.
* Calibration behaviour may depend on model architecture, environment, dataset, and random seed.
* Multi-step uncertainty propagation is not the primary thesis-level outcome.
* Mechanism comparisons require careful control or matching of intervention severity.

These limitations will be refined as the experimental evidence develops.

---

## Reproducibility

The repository is designed around **explicit configuration, controlled randomness, and traceable outputs**.

Each experiment should record:

* environment
* dataset version / manifest
* model configuration
* random seed
* shift mechanism
* intervention severity
* output location

The intended principle is:

> **Every reported result should be reproducible from code, configuration, and recorded experimental metadata.**

Large datasets remain outside Git; only manifests and relevant reproducibility metadata belong in the repository.

---

## Repository Structure

```text
causal-calibration-attribution/
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
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── notebooks/
│
├── docs/
│   ├── research_questions.md
│   ├── methodology.md
│   ├── counterfactual_design.md
│   ├── causal_assumptions.md
│   ├── limitations.md
│   └── related_work.md
│
└── tests/
```

### Design principles

The repository architecture follows the scientific structure of the project:

* `models/` — probabilistic world models and ensembles
* `shifts/` — dynamics, observation, and policy interventions
* `metrics/` — calibration and uncertainty metrics
* `causal/` — counterfactual and causal analysis
* `stats/` — effect estimation and statistical inference
* `experiments/` — reproducible experiment definitions and execution
* `results/` — machine-readable outputs, tables, figures, and logs
* `docs/` — methodological and scientific documentation
* `tests/` — implementation and statistical checks

The `causal/` layer is explicit because causal attribution is a **core methodological component** of the thesis rather than an analysis hidden inside notebooks.

---

## Research Scope

This repository represents the **MSc thesis / Paper 1 diagnosis stage** of a broader research program:

**Thesis → Paper 1 → Paper 2 → PhD**

### MSc Thesis

**Counterfactual Causal Attribution of Calibration Failure in Uncertainty-Aware World Models under Mechanism-Specific Distribution Shift**

Scientific foundation: controlled mechanism-specific interventions and causal / counterfactual attribution of calibration failure.

### Paper 1

**Why Ensembles Miscalibrate Differently: Causal Attribution of Calibration Failure under Dynamics, Observation, and Policy Shift**

Focus: empirical diagnosis and mechanism-specific comparison.

### Later Research

Future work may extend the diagnosis toward:

**Causally-Informed Recalibration → Safety-Constrained Decision Making → Reliable AI Decision-Making under Distribution Shift**

These later components are not represented here as completed contributions.

---

## Author

**Morteza Toghani**
MSc Artificial Intelligence & Robotics

---

## Research Identity

> **Reliable Decision-Making for AI Systems under Uncertainty and Distribution Shift: A Causal Approach to Calibration and Safety**

**Research umbrella:**
Reliable AI Decision-Making under Uncertainty and Distribution Shift
