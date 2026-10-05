# Component status (blueprint evidence taxonomy)

"In this repository" matters: several results were produced by the author's earlier pipeline whose code has
not been migrated yet.

| Component | Status in this repository | Notes |
|---|---|---|
| Calibration metrics (`src/calibration/scalar_metrics.py`) | PROTOTYPE | unit-tested on synthetic data; reproduces stored calibration errors exactly; sharpness/NLL/CRPS not compared to stored values |
| Ensemble aggregation (`src/model/aggregation.py`) | PROTOTYPE | aggregation + interface only; no trainable model |
| World-model training (PyTorch ensemble) | NOT STARTED here | exists in the author's pipeline; to be migrated |
| Observation shift (`src/interventions/observation.py`) | PROTOTYPE | reference scale not verified against original |
| Dynamics shift | NOT STARTED here | results exist from the original pipeline |
| Policy shift | NOT STARTED here | results exist from the original pipeline |
| Episode splitter (`src/data/splits.py`) | PROTOTYPE | not used to regenerate the original split |
| Causal estimator (`src/causal/attribution.py`) | PROTOTYPE | paired differences and seed-level effects |
| Statistics (`src/stats/`) | PROTOTYPE | seed-level bootstrap, t, sign-flip, Holm, d_z |
| Hopper 5-seed mechanism results | PRELIMINARY | one environment, n = 5, severity unmatched |
| Walker2d replication | NOT STARTED | environment smoke test only |
| CRN-paired evaluation and intervention audits | NOT STARTED | |
| Recalibration (`src/recalibration/`) | PLANNED | Paper 2, private; folder not created |
| Safety layer | PLANNED | Paper 2, private |
| CI workflow | IMPLEMENTING | file present; not yet run on GitHub |

## Migration checklist (from the author's pipeline)
1. Model training code and the exact baseline configuration; freeze it on development data.
2. Dynamics intervention code and the physical parameters behind "low"/"high".
3. Policy intervention code and the new-policy definition.
4. Definition of the pair unit behind the reported pair-level ATE.
5. Observation-noise reference scale.
6. A matched-severity protocol, then CRN-paired evaluation and audits.
7. Walker2d replication.
