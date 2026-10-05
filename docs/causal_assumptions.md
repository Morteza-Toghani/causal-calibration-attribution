# Causal assumptions

| # | Assumption | Why it matters | How it is (or will be) checked | Status |
|---|---|---|---|---|
| A1 | Each intervention changes only its target mechanism (mechanism-specificity) | otherwise the effect cannot be attributed | intervention-validation checks, incl. cross-effects (e.g. does P also change the transition distribution?) | NOT STARTED |
| A2 | Baseline and treatment share all controllable randomness (CRN) | otherwise sampling noise enters the paired difference | unit test for reuse (`tests/test_shifts.py` covers the observation noise); audit in pipeline | PARTIAL |
| A3 | Intensities are comparable across mechanisms (matched severity) | otherwise "mechanism effect" may be a severity effect | severity definition and matching protocol | NOT STARTED |
| A4 | Same trained model across factual and counterfactual condition | isolates evaluation-time shift from training variability | by design: interventions are applied at inference, no retraining | BY DESIGN |
| A5 | Pairs are exchangeable units for the reported pair-level ATE | needed for pair-level CIs and p-values | pairs come from one model; treat as descriptive; inference is done over seeds | KNOWN LIMITATION |
| A6 | Probe states for P are fixed before action sampling | avoids state-distribution drift confounding | audit in pipeline | NOT STARTED |
| A7 | Observation noise does not leak the prediction target | otherwise corrupted inputs encode `s_{t+1}` | noise audit | NOT STARTED |

Confounders and controls named by the blueprint: severity, model capacity, seed, task conditions.

This document lists assumptions; it does not claim they hold.
