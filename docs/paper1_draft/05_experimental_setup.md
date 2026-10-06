# 5. Experimental Setup

## 5.1 Environments and datasets

We used two MuJoCo locomotion environments with offline data from the Minari library.

| Environment | Minari ID | Episodes | Steps | obs_dim | action_dim |
|---|---|---|---|---|---|
| Hopper | `mujoco/hopper/medium-v0` | 1327 | 999,404 | 11 | 3 |
| Walker2d | `mujoco/walker2d/medium-v0` | 1044 | 999,613 | 17 | 6 |

Dataset names, versions, episode boundaries, and metadata were recorded in a manifest committed with the code. Minari 0.5.3, MuJoCo 3.2.3, Gymnasium 1.3.0. Both datasets are of the "medium" quality level. We did not use replay or expert datasets, and we did not test other quality levels.

## 5.2 Model training

The architecture is described in Section 4.2. Hyperparameters were frozen before the final experiments and were identical across both environments.

| Hyperparameter | Value |
|---|---|
| Ensemble members | 5 |
| Hidden layers | (256, 256), ReLU |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-5 |
| Batch size | 1024 |
| Epochs | 25 |
| Gradient clipping (max norm) | 5.0 |
| Log-variance clamp | [−10.0, 5.0] |
| Standardization | per-dimension mean/std, fitted on train split |
| Loss | Gaussian NLL, mean over batch and dimensions |

We did not tune hyperparameters per environment.

## 5.3 Data splits

Splits were episode-level and deterministic, with fractions of 70% train, 10% calibration, 10% test, and 10% probe. The split seed was 1000 plus the training-seed index, so each training seed used a different partition of episodes. Transitions were capped at 150k (train), 25k (calibration), 25k (test), and 10k (probe).

## 5.4 Intervention parameters

| Mechanism | Parameter | Low | High | Target |
|---|---|---|---|---|
| Observation (O) | noise fraction $f$ | 0.05 | 0.10 | dataset target (unchanged) |
| Dynamics (D) | body-mass scale $c$ | 0.95 | 0.80 | re-simulated |
| Policy (P) | action scale $\kappa$ | 0.95 | 0.80 | re-simulated |

Identity controls used $c = 1.0$ and $\kappa = 1.0$. Severities were not matched across mechanisms: a mass scale of 0.80, a noise fraction of 0.10, and an action scale of 0.80 are not equivalent in any common unit. We did not run a pilot to equate severities, and so the comparison between mechanisms is a comparison of effects at the chosen settings.

## 5.5 Training seeds

| Environment | Seeds | n |
|---|---|---|
| Hopper | 0–4 (legacy pipeline) and 5–9 (retrained with identical hyperparameters) | 10 |
| Walker2d | 0–4 | 5 |

Hopper seeds 0–4 came from the original pipeline, and seeds 5–9 were trained later with the reimplementation under identical hyperparameters. We did not test whether the two Hopper groups differ systematically. The baseline signed-coverage analysis (Section 6.2) uses the five legacy seeds only.

## 5.6 Reproducibility

Training and evaluation were run on CPU only (12 physical cores; PyTorch restricted to 6 threads) under Windows. PyTorch 2.7.1+cpu, NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.1, Matplotlib 3.10.9.. Random generators were seeded per training seed and per evaluation. We did not assume bit-level determinism of GPU operations. No determinism flags beyond per-seed and per-member seeding were enabled. All reported bit-exact reproductions were obtained on CPU. Model checkpoints, configuration files, and run manifests were committed, and every table in this paper is generated from machine-readable result files, not entered by hand. As a validation of the reimplementation, all five legacy Hopper seeds reproduced the original pipeline's `baseline_metrics.json` with a maximum absolute difference below 1.2 × 10⁻¹⁶ across seven metrics (Section 6.1). https://github.com/Morteza-Toghani/causal-calibration-attribution (commit 6a17e80; archive DOI to be assigned at submission).
