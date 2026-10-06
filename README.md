# Counterfactual Causal Attribution of Calibration Failure in Uncertainty-Aware World Models

> **MSc Thesis Research · Artificial Intelligence & Robotics**

This repository contains the research code and experimental design for my Master's thesis on calibration failure in uncertainty-aware world models under mechanism-specific distribution shift.

The project asks a simple question: when the data distribution changes, **which part of the change is responsible for the resulting calibration failure, and how large is that effect?**

The study focuses on three mechanisms of distribution shift:

* **Dynamics shift**
* **Observation shift**
* **Policy shift**

The main goal is to compare these mechanisms under controlled conditions rather than treating distribution shift as a single undifferentiated source of failure.

---

## Why this problem?

Uncertainty estimates are useful only when they provide a reasonably calibrated description of predictive uncertainty.

Under distribution shift, this relationship can break down. A model may remain useful in some respects while its uncertainty estimates become less reliable. More importantly, different types of distribution shift do not necessarily affect the predictive pipeline in the same way.

A change in transition dynamics, for example, is different from corrupting the observation process. A policy change is different again, because it can alter which states and actions are being evaluated.

This thesis studies these mechanisms separately and asks whether their effects on calibration can be distinguished in a controlled experimental setting.

---

## Research Question

> **Which mechanism-specific distribution shift is associated with calibration failure in an uncertainty-aware offline world model, and by how much under controlled intervention?**

The primary comparison is between:

1. **Dynamics intervention**
2. **Observation intervention**
3. **Policy intervention**

The aim is not simply to compare which condition has the largest error. The aim is to examine whether changing the shift mechanism produces a systematic change in calibration-related outcomes under a controlled design.

---

## Study Design

The basic unit of comparison is a matched model-evaluation instance.

The same trained model is evaluated under a baseline condition and under a controlled intervention. Wherever possible, the same evaluation states and reusable sources of randomness are retained across the paired conditions.

This allows the analysis to focus on the difference introduced by the intervention rather than on unrelated variation between runs.

The thesis uses **Common Random Numbers (CRN)** and paired comparisons as part of this design.

The primary estimand is a mechanism-specific average treatment effect relative to baseline. In practical terms, this means measuring how much the chosen calibration outcome changes when one shift mechanism is introduced while the rest of the comparison is kept controlled.

The interpretation remains tied to the experimental design and its assumptions. The project does not claim that a controlled simulation automatically establishes a causal effect in the real world.

---

## Shift Mechanisms

### Dynamics shift

A controlled change is introduced into the transition dynamics during evaluation.

The observation process and action-selection mechanism are kept fixed for the main comparison.

### Observation shift

A controlled transformation is applied to the observations, such as added noise or corruption.

The underlying dynamics and policy are kept fixed.

### Policy shift

The action-selection rule is changed while evaluating the model on a fixed set of probe states.

The main analysis does not use unrestricted rollouts for this intervention. This is intended to avoid mixing the effect of policy change with additional state-distribution drift caused by a new rollout.

The thesis treats these three mechanisms as distinct interventions because they affect different parts of the predictive pipeline.

---

## World Model

The project uses a **state-based probabilistic ensemble world model**.

The baseline design is a small ensemble of multilayer perceptrons trained on offline reinforcement-learning data.

**Model configuration**

* MLP-based probabilistic predictors
* **5 ensemble members**
* Next-state prediction (1-step)
* Gaussian predictive distributions
* Ensemble-based uncertainty estimation

Each ensemble member predicts the next state as a probability distribution conditioned on the current state and action.

In practical terms, the model predicts both a mean and a variance for its next-state estimate. The predictions from the ensemble are then combined to estimate predictive uncertainty and ensemble disagreement.

The ensemble is used as a practical uncertainty-estimation method rather than as the scientific contribution by itself.

---

## Data

The study is designed around **offline reinforcement-learning data** with explicit dataset metadata and manifests.

The current research plan prioritizes benchmark datasets compatible with the Minari / RL4D ecosystem, with **Hopper** and **Walker2d** as the main environments.

Large raw datasets will remain outside Git. The repository will contain dataset manifests, version information, and the metadata needed to reconstruct the experimental setup.

---

## Prediction Horizon

The primary analysis uses **1-step transition prediction**.

This keeps the attribution problem as direct as possible: the intervention is applied to one transition and its effect is measured on the corresponding predictive uncertainty and calibration outcome.

Multi-step rollouts may be studied later as a robustness or propagation analysis, but they are not the primary thesis-level outcome.

---

## Calibration and Evaluation

The main outcome is a **regression-specific calibration error** appropriate for continuous probabilistic prediction.

The evaluation also considers:

* **Negative Log-Likelihood (NLL)**
* **Prediction-interval coverage** (at nominal 50% and 90%)
* **Sharpness**
* **CRPS**
* Predictive uncertainty
* Ensemble disagreement
* Calibration curves
* Error-uncertainty relationships

Calibration and sharpness are treated as complementary properties. A useful uncertainty estimate should not only be statistically aligned with observed outcomes, but should also remain informative rather than unnecessarily broad.

The project therefore does not rely on the classical classification ECE as the primary uncertainty-calibration measure for continuous regression.

---

## Statistical Analysis

The comparison is designed around paired observations and repeated model instances.

The planned analysis includes:

* paired differences (ATE)
* bootstrap confidence intervals
* paired permutation / sign-flip tests
* effect sizes
* multi-seed evaluation
* Holm correction for multiple comparisons

The emphasis is on estimating the size and uncertainty of the mechanism-specific effect, rather than reporting statistical significance alone.

The current design recommends multiple matched training seeds, with the exact number adjusted according to the available compute budget and pilot results.

---
## Key findings

After migrating the original Paper 1 pipeline to this modular repository
and reproducing its baseline and all three interventions **bit-for-bit**
across 5 seeds, we obtained the following results. All numbers are
means over 5 seeds; confidence intervals use Student's t with `df=4`
(`t_crit = 2.776`).

### 1. Baseline over-dispersion (α ≈ 0.24)

The 5-member ensemble **over-covers at every nominal level**. Signed
coverage at nominal 0.50 is `+0.35` (empirical 0.85 instead of 0.50).
A single per-seed variance scaling factor `alpha ≈ 0.24` fitted on the
**calibration split** (nominal 0.90) brings coverage to nominal. The
baseline predictive variance is roughly 4× too large.

### 2. Observation shift dominates the mechanism story

After recalibration, the effect of mechanism-specific shift on the
regression calibration error is:

| Condition | ATE | 95% t-CI | Magnitude |
|---|---|---|---|
| `observation_high` | **+0.264** | [+0.261, +0.267] | ×70 |
| `observation_low`  | **+0.199** | [+0.194, +0.203] | ×53 |
| `dynamics_high`    | -0.0038 | [-0.0059, -0.0017] | ×1.0 |
| `policy_high`      | +0.0027 | [+0.0018, +0.0037] | ×0.7 |
| `dynamics_low`     | -0.0012 | [-0.0016, -0.0008] | ×0.3 |
| `policy_low`       | +0.0008 | [+0.0004, +0.0012] | ×0.2 |

The observation pathway produces effects **two orders of magnitude
larger** than dynamics or policy. Once the baseline is properly
calibrated, only observation shift has a practically relevant effect.

### 3. Re-simulation bias in dynamics and policy

An identity intervention (`mass_scale=1.0` for dynamics, `action_scale=1.0`
for policy) does **not** reproduce the dataset target bit-for-bit:
`max|y_new - y| ≈ 2.7` and ATE bias ≈ -0.003 per seed. The cause is
MuJoCo `set_state` + `step` not resetting `qacc_warmstart` internally,
plus the x=0 reconstruction of state from the observation. Correcting
the paired baseline to use `y_resim(scale=1.0)` changes the D and P
ATEs substantially — see [`docs/paper1_findings.md`](docs/paper1_findings.md).

### Reproduce

```bash
python scripts/reproduce_baseline_all_seeds.py         # baseline, 5 seeds
python scripts/reproduce_interventions_all_seeds.py    # interventions, 5 seeds
python scripts/null_control.py                         # re-simulation bias
python scripts/analyze_signed_coverage.py              # direction-aware coverage
python scripts/analyze_interventions_corrected.py      # corrected ATE
python scripts/analyze_t_ci.py                         # t-CI corrected
python scripts/recalibrate_and_rerun.py                # recalibration
python scripts/analyze_t_ci_recalibrated.py            # t-CI recalibrated



## Current Status

> **Status: IN DESIGN / PILOT PHASE**

The research question and experimental protocol are defined. The implementation is partially complete, and the full pipeline (real data, multi-seed, mechanism-specific interventions, CRN-paired evaluation) is still in progress.

The project follows an evidence-based maturity model:

`IN DESIGN -> PROTOTYPE -> PRELIMINARY -> VALIDATED -> REPRODUCIBLE`

These labels are intended to reflect the actual state of the work. A running script is not treated as validated evidence.

No final empirical claim is reported in this repository until the corresponding experiment, analysis, and reproducibility checks have been completed.

---

## Repository Structure

```text
causal-calibration-attribution/
|
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- environment.yml
|-- pyproject.toml
|
|-- configs/
|   |-- model/
|   |-- shift/
|   +-- experiment/
|
|-- data/
|   +-- manifests/
|
|-- src/
|   |-- data/
|   |-- model/
|   |-- calibration/
|   |-- interventions/
|   |-- causal/
|   +-- stats/
|
|-- experiments/
|   |-- diagnostics/
|   |-- pilot/
|   |-- phase_A_baseline_calibration/
|   +-- phase_B_mechanism_interventions/
|
|-- results/
|   |-- raw/
|   |-- tables/
|   +-- figures/
|
|-- notebooks/
|
|-- docs/
|   |-- research_questions.md
|   |-- methodology.md
|   |-- counterfactual_design.md
|   |-- causal_assumptions.md
|   |-- limitations.md
|   |-- status.md
|   |-- decision-log.md
|   |-- evidence-map.md
|   |-- negative-results.md
|   |-- provenance.md
|   +-- related_work.md
|
|-- scripts/
+-- tests/
```

The main directories have distinct roles:

* `src/data/` contains dataset loading and episode-level splitting.
* `src/model/` contains the probabilistic world-model implementation.
* `src/calibration/` contains calibration and uncertainty metrics.
* `src/interventions/` contains the dynamics, observation, and policy interventions.
* `src/causal/` contains the counterfactual and causal-analysis layer.
* `src/stats/` contains effect estimation and statistical inference.
* `experiments/` contains experiment definitions and execution scripts.
* `results/` contains figures, tables, and machine-readable outputs.
* `docs/` contains the methodological documentation.
* `tests/` contains implementation and statistical checks.

The causal layer is kept explicit because causal attribution is part of the scientific method of the thesis, not an analysis hidden inside a notebook.

---

## Research Lineage

This repository is the **MSc thesis / diagnosis stage** of a broader research program.

### MSc Thesis

**Counterfactual Causal Attribution of Calibration Failure in Uncertainty-Aware World Models under Mechanism-Specific Distribution Shift**

The thesis focuses on identifying and quantifying mechanism-specific calibration failure under controlled interventions.

### Paper 1

**Why Ensembles Miscalibrate Differently: Causal Attribution of Calibration Failure under Dynamics, Observation, and Policy Shift**

The planned paper extends the empirical diagnosis and studies mechanism differences in greater detail.

### Broader Research Direction

**Reliable Decision-Making for AI Systems under Uncertainty and Distribution Shift: A Causal Approach to Calibration and Safety**

The longer-term research direction moves from diagnosis toward causally informed recalibration and safer downstream decision-making.

---

## Author

**Morteza Toghani**

MSc Artificial Intelligence & Robotics
