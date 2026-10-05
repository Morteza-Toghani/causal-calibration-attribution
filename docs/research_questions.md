# Research questions and hypotheses

Source of truth: the GitHub Master Blueprint (Layers 3-4). Titles are locked; see README.

| ID | Question | Hypothesis | Where addressed | Status |
|---|---|---|---|---|
| RQ1 | Does calibration degrade differently under dynamics, observation and policy shift? | H1 mechanism-specific miscalibration | `experiments/phase_B_*`, `results/` | PRELIMINARY (Hopper, 5 seeds) |
| RQ2 | Is the failure attributable to the mechanism rather than to aggregate OOD severity? | - | needs matched severity (see `counterfactual_design.md`) | IN DESIGN |
| RQ3 | Under matched conditions, what would the calibration outcome have been under an alternative mechanism? | H2 counterfactual mechanism effect | `src/causal/` (estimator only) | IN DESIGN |
| RQ4 | Are mechanism-specific effects robust across seeds and repetitions? | H3 multi-seed robustness | `src/stats/`, `scripts/analyze_results.py` | PRELIMINARY (n = 5 seeds) |
| RQ5 | Can mechanism-informed recalibration beat a shift-agnostic baseline? | H4 | not in this repository | PLANNED (Paper 2, private) |
| RQ6 | Can mechanism-informed uncertainty correction improve safety without excess conservatism? | H5 | not in this repository | PLANNED (Paper 2, private) |

## How the thesis-level questions (RQ0-RQ7 of the Master Research Plan) map here

| Master plan | This table |
|---|---|
| RQ1 dynamics, RQ2 observation, RQ3 policy, RQ4 comparative attribution | RQ1, RQ2 |
| RQ5 intensity | covered by the low/high design inside RQ1 |
| RQ6 reliability of uncertainty (NLL, coverage, sharpness) | reported next to calibration error in RQ1 |
| RQ7 generalisation to a second environment | Walker2d replication - NOT STARTED |

## H1-H5 wording discipline

H2 is *not* "mechanism A happened to give a lower score". It is: under the defined intervention design,
changing the mechanism is associated with a systematic change in the outcome. Language in this repository
follows the maturity rule in the README (no "demonstrates" / "establishes" before the evidence exists).
