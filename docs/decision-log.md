# Decision log

Entries record reasoning, not just outcomes. Decisions #001-#007 come from the project plan / pilot work;
#008-#010 were taken when this repository was set up (October 2026) and should be confirmed by the author.

### Decision #001
Question: Why use mechanism-specific interventions rather than one generic OOD severity?
Decision: Separate dynamics, observation and policy shift.
Alternatives: One aggregate OOD score.
Reason: Aggregate severity cannot identify which shift pathway generated the observed calibration failure.
Evidence: project plan; not yet tested against an aggregate-severity baseline.
Consequences: Three distinct intervention implementations are required; severity matching becomes necessary (#007).
Status: Active.

### Decision #002
Question: Which prediction horizon carries the primary analysis?
Decision: One-step transition prediction.
Alternatives: Multi-step rollouts from the start.
Reason: Compounding error and policy-visitation feedback would confound attribution.
Consequences: Multi-step is a Level-B extension only.
Status: Active.

### Decision #003
Question: Retrain per intervention or intervene at evaluation time?
Decision: Train once per seed; apply interventions at evaluation time.
Alternatives: Train a model per condition.
Reason: Keeps the model fixed across factual/counterfactual conditions and reduces compute.
Consequences: Shifts are evaluation-time perturbations, not training-distribution shifts.
Status: Active.

### Decision #004
Question: Which calibration metric?
Decision: Regression-specific interval-coverage calibration error (not classification ECE / sklearn `calibration_curve`).
Alternatives: Binned ECE.
Reason: Targets are continuous; the metric must be defined on predictive intervals.
Evidence: definition reproduces stored `calibration_error` values exactly (`scripts/analyze_results.py`).
Status: Active.

### Decision #005
Question: Is more training the fix for over-coverage?
Decision: No; the variance floor was lowered instead (min_log_var -10 -> -20) and remains under study.
Alternatives: 50 epochs at min_log_var -10 (config B).
Reason: Config B improved NLL but worsened calibration (`negative-results.md`, N1).
Evidence: single-seed pilot, test-set diagnostics (see limitation 8).
Consequences: Baseline not yet frozen; configs A-D remain candidates.
Status: Open.

### Decision #006
Question: How is the final evaluation protected from selection effects?
Decision: Development split for configuration selection; test split reserved for the final frozen evaluation.
Reason: The test split was already used for diagnostics during the pilot.
Consequences: Final claims require a fresh protocol; see `methodology.md`.
Status: Active (not yet satisfied).

### Decision #007
Question: Can mechanisms be ranked from the current results?
Decision: No. Mechanisms are compared only under a documented matched-severity protocol, which does not exist yet.
Reason: Intensities are not on a common scale and pair counts differ across mechanisms.
Status: Active.

### Decision #008
Question: What do phases A-F mean?
Decision: A baseline; B interventions; C CRN-paired evaluation and audits; D causal statistics; E replication; F Level-B extension.
Alternatives: Create empty `phase_C..F` folders as in the blueprint tree.
Reason: Empty folders signal progress that does not exist (blueprint golden rule 1 / anti-amateur audit #5).
Consequences: Only `phase_A` and `phase_B` exist on disk; C-F are listed in the README table as NOT STARTED.
Status: Proposed - confirm.

### Decision #009
Question: What is the inference unit for the reported statistics?
Decision: The seed. The per-pair CIs and p-values in the original CSV are kept as raw inputs but not used for conclusions.
Alternatives: Treat the 5,000-25,000 evaluation pairs as independent.
Reason: Pairs come from one model per seed and are not independent replicates of the training procedure.
Consequences: n = 5 and a minimum attainable p of 0.0625 (`theoretical_analysis.md`).
Status: Proposed - confirm.

### Decision #010
Question: Which blueprint components are not created yet?
Decision: `configs/model/rff_ensemble.yaml`, `src/recalibration/`, and `phase_C..F` are not created.
Reason: The RFF ensemble is not part of the current plan, and recalibration belongs to Paper 2 (private); Rule 5 forbids listing unused technology.
Status: Proposed - confirm.
