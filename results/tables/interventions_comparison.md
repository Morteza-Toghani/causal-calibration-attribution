# Interventions - original vs corrected

Correction: for D and P, baseline target is now y_resim(scale=1.0)
instead of y_dataset. This removes the re-simulation bias.

| mechanism | intensity | ATE (original) | ATE (corrected) | SD | re-sim bias |
|---|---|---|---|---|---|
| dynamics | high | -0.02481 | -0.02163 | 0.00254 | -0.00318 |
| dynamics | low | -0.00593 | -0.00275 | 0.00076 | -0.00318 |
| observation | high | +0.04249 | +0.04249 | 0.01016 | +0.00000 |
| observation | low | -0.05878 | -0.05878 | 0.01074 | +0.00000 |
| policy | high | +0.00335 | +0.00658 | 0.00253 | -0.00324 |
| policy | low | +0.00024 | +0.00347 | 0.00052 | -0.00324 |