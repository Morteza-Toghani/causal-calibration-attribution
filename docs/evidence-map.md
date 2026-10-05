# Evidence map (claim -> code -> result)

Chain per claim: Claim -> RQ -> Hypothesis -> Causal assumption -> Intervention -> Counterfactual condition ->
Experiment -> Code -> Configuration -> Statistical analysis -> Result -> Status.

| # | Claim | RQ / H | Assumption | Intervention / counterfactual | Experiment | Code | Config | Statistics | Result | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | The baseline ensemble over-covers (intervals too wide) | RQ1 / H1 | - | none (factual) | phase A / B baseline rows | `scripts/analyze_results.py`, `src/calibration/scalar_metrics.py` | `configs/experiment/phase_A.yaml` | mean and SD over 5 seeds | `results/tables/aggregate_metrics_mean_over_seeds.md` | PRELIMINARY |
| C2 | Observation shift (noise fraction 0.05, 0.10) reverses coverage from over- to under-coverage | RQ1 / H1 | A1, A2 | observation / same instance, rescaled base noise | phase B | same + `src/interventions/observation.py` (prototype; not the original implementation) | `configs/shift/observation.yaml` | seed-level summary, signed bias | `results/tables/mechanism_attribution_summary.md`, `results/figures/fig_calibration_curves.png` | PRELIMINARY |
| C3 | Dynamics shift slightly reduces coverage | RQ1 / H1 | A1 | dynamics / not in repository | phase B | `scripts/analyze_results.py` (analysis only) | `configs/shift/dynamics.yaml` (parameters open) | seed-level summary | `results/tables/mechanism_attribution_summary.md` | PRELIMINARY |
| C4 | Policy shift on fixed probe states has near-zero effect | RQ1 / H1 | A6 | policy / not in repository | phase B | `scripts/analyze_results.py` (analysis only) | `configs/shift/policy.yaml` (parameters open) | seed-level summary | `results/tables/mechanism_attribution_summary.md` | PRELIMINARY |
| C5 | Effects are directionally consistent across seeds | RQ4 / H3 | - | - | phase B | `src/stats/multi_seed_runner.py` | - | sign count, d_z, bootstrap CI | `results/figures/fig_mechanism_attribution_forest.png` | PRELIMINARY |
| C6 | A mechanism ranking is possible | RQ2, RQ3 / H2 | A3 | - | - | - | - | - | none | NOT SUPPORTED (severity not matched) |
| C7 | Recalibration / safety improvements | RQ5, RQ6 / H4, H5 | - | - | - | - | - | - | none | PLANNED (Paper 2, private) |

Rows C6-C7 are kept on purpose so that unsupported claims stay visibly unsupported.
