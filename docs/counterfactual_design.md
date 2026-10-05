# Counterfactual design

Unit: a matched model-evaluation instance - the same trained model, seed, evaluation data or probe
states, and reusable randomness appear in both the factual and the intervened condition.
A shared seed alone is not enough; the pipeline must show the corresponding random draws are reused.

Estimand (controlled simulation): `ATE_m = E[Y(m) - Y(0)]`, estimated by
`mean_i (Y_i(m) - Y_i(0))` (`src/causal/attribution.py`). `Y` = calibration error (primary) or a secondary metric.

| | Dynamics (D) | Observation (O) | Policy (P) |
|---|---|---|---|
| Factual condition | baseline simulator, data policy, clean observations | same | same |
| Intervention | controlled change of simulator transition parameters (e.g. mass, friction) at inference | additive Gaussian corruption `o' = o + eps` | change of the action-selection rule on **fixed probe states**; no free rollout |
| Counterfactual condition | same instance with the changed parameters | same instance with the same base noise rescaled | same probe states with new-policy actions |
| Held fixed | observation, action selection | dynamics, action selection | dynamics, observation, probe states |
| Reused randomness | initial states, evaluation draws | base noise `eps`, evaluation draws | probe states, sampling draws |
| Outcome | calibration error of the one-step predictive distribution | same | same |
| Intensities | low / high - parameters **not yet migrated** | noise fraction 0.05 / 0.10 of a reference scale (reference scale to be verified) | low / high - **not yet migrated** |
| Implementation here | none | `src/interventions/observation.py` (prototype) | none |

## Interpretation
Because the experimenter applies the intervention in simulation, the contrast is an *experimental* effect
of the intervention as implemented. Reading it as the effect of "the mechanism" additionally requires
that the intervention is mechanism-specific and that mechanisms are compared at comparable severity
(`causal_assumptions.md`).

## Matched-severity requirement (not yet satisfied)
A mechanism ranking is only meaningful once severity is defined on a common scale. Required and
currently missing: severity definition, severity parameterisation, matching approach, residual
differences. In the Phase-B results the number of evaluation pairs differs by mechanism (5,000 / 25,000 /
10,000 for D / O / P) and the intensities are not matched on any shared scale, so mechanisms must not be
ranked from those results.

## Limitations of this design
See `limitations.md`.
