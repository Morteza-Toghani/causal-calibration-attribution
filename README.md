# Counterfactual Causal Attribution of Calibration Failure in Uncertainty-Aware World Models

Master's thesis project investigating which mechanism-specific distribution shift (dynamics, observation, or policy) causally produces calibration failure in an uncertainty-aware offline world model, and by how much.

## Research Question

Which mechanism-specific distribution shift causally produces calibration failure in an uncertainty-aware offline world model, and by how much?

## Approach

- State-based ensemble world model (MLP, K=5 members) trained on offline RL data
- Counterfactual interventions applied at inference time: dynamics, observation, policy
- Common Random Numbers (CRN) + paired differences for causal comparison
- Bootstrap confidence intervals and paired permutation tests for statistical validation

## Repository Structure
configs/         # environment, dataset, seed, intervention configs
data/             # dataset manifests (raw data kept outside Git)
src/model/        # probabilistic MLP + ensemble
src/interventions/# dynamics, observation, policy interventions
src/calibration/  # calibration error, coverage, sharpness, CRPS metrics
src/stats/        # ATE estimation, bootstrap, permutation tests, Holm correction
experiments/      # run scripts
results/          # tables, figures, machine-readable outputs
notebooks/        # exploratory analysis only
## Status

🚧 Work in progress — pilot phase.

## Author

Morteza Toghani — MSc Artificial Intelligence & Robotics
