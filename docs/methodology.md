# Methodology

## World model
Probabilistic one-step model of `s_{t+1}` given `(s_t, a_t)`: an ensemble of `K = 5` Gaussian MLPs, each
outputting mean and log-variance per state dimension, trained with Gaussian NLL
(`configs/model/mlp_ensemble.yaml`). Ensemble moments (implemented and tested in
`src/model/aggregation.py`):

```
mean = mean_k mu_k      aleatoric = mean_k sigma_k^2      epistemic = Var_k mu_k      total = aleatoric + epistemic
```

The trained PyTorch ensemble is **not yet in this repository** (see `status.md`).

## Why one step
The primary analysis is one-step prediction: the intervention and the outcome act on the same
transition, which keeps attribution direct. Multi-step rollouts add compounding error and
policy-visitation feedback and are a Level-B extension.

## Data
Minari `mujoco/hopper/medium-v0`, split **by episode** (no transition-level leakage):
train 929 / development 199 / test 133 / probe 66 episodes (999,404 transitions in total; counts
from the project report, encoded in `configs/experiment/phase_A.yaml` and checked by `tests/test_configs.py`).
`src/data/splits.py` provides a generic, tested episode-level splitter; it has **not** been used to
regenerate the original split.

The "development" split was used for configuration choices (e.g. validation NLL); the test split was
inspected for diagnostics during pilot work and therefore can no longer serve as an untouched
hold-out for configuration selection. Final claims need a protocol where configuration is frozen on
development data before the final test evaluation.

## Interventions (applied at evaluation time, no retraining)
Dynamics, observation and policy shift; one mechanism changes at a time. Definitions, controls and
assumptions: `counterfactual_design.md`, `causal_assumptions.md`.

## Outcomes
| Metric | Role | Definition |
|---|---|---|
| calibration error | primary | mean over nominal levels 0.1...0.9 of `abs(pooled empirical coverage - nominal)`, central Gaussian intervals |
| signed calibration bias | diagnostic | same without the absolute value (>0 over-coverage) |
| NLL | secondary | mean per-element Gaussian NLL |
| coverage@50 / @90 | secondary | pooled empirical coverage |
| sharpness | secondary | mean interval width |
| CRPS | secondary | closed-form Gaussian CRPS |

The primary definition reproduces the stored `calibration_error` of every seed and condition in
`results/raw` exactly (asserted in `scripts/analyze_results.py`). Sharpness, NLL and CRPS in `results/raw`
were not recomputed here because the raw predictions are not in this repository.

## Statistical analysis
Inference unit: the **seed** (one trained ensemble). Per condition: mean effect, SD across seeds, 95%
percentile-bootstrap CI over seeds (10,000 resamples) and Student-t CI, Cohen's d_z, number of seeds
with a positive effect, exact two-sided paired sign-flip permutation p-value, Holm correction over the
six conditions. With n = 5 the smallest attainable exact two-sided p is 2/2^5 = 0.0625.

## Phases
| Phase | Content | Master-plan pipeline stage | Status |
|---|---|---|---|
| A | baseline calibration and diagnostics | 3 | PRELIMINARY |
| B | dynamics / observation / policy interventions | 4-6 | PRELIMINARY (results only; code not migrated) |
| C | CRN-paired evaluation and sanity audits | 7 | NOT STARTED |
| D | causal statistics (ATE, bootstrap, permutation, Holm) with matched severity | 8 | IN DESIGN |
| E | second-environment replication (Walker2d) | 9 | NOT STARTED |
| F | Level-B extension: factorial / multi-step / ensemble-size | - | NOT STARTED |

The letter-to-stage mapping is a repository-level convention (decision #008), not part of the original plan.
