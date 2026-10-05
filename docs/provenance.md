# Results Provenance and Clean-Rerun Policy

## Current repository evidence

The repository contains five Hopper seed summaries under `results/raw/hopper_seeds_0_4/` with baseline and six intervention conditions. The raw files are hash-checked and the analysis tables are reproducibly rebuilt from them.

## What is currently verified

- Environment: Hopper
- Seeds: 0-4
- Raw result integrity: verified by SHA256
- Analysis-table regeneration: verified
- Intervention parameter values present in raw summaries:
  - observation noise_fraction: 0.05 / 0.10
  - policy action_scale: 0.95 / 0.80
  - dynamics mass_scale: 0.95 / 0.80

## What is NOT verified

The training/intervention pipeline that originally produced the five-seed raw results is not present in this repository. Therefore the exact original training configuration (for example `min_log_var`, epochs, learning rate, hidden dimensions, and exact split usage) cannot be reconstructed from the repository alone.

These existing results remain **PRELIMINARY / provenance-incomplete** and must not be used as the final frozen causal result.

## Clean rerun policy

The final thesis evidence will be regenerated from versioned code and configs using this protocol:

1. Development split selects the baseline configuration.
2. The baseline configuration is frozen before final test evaluation.
3. The final test split is treated as a holdout after freeze.
4. The same trained model/seed/probe data and reusable randomness are used for paired counterfactual evaluation.
5. Every run stores machine-readable metadata including repository commit, config, seed, environment, split, and timestamp.

