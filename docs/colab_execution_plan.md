# Colab Execution Plan

Google Colab is the compute environment. GitHub remains the source of truth for code, configuration, documentation, tests, and lightweight reproducible results.

## Principle

`GitHub -> clone -> Colab execution -> save artifacts -> sync lightweight outputs back to GitHub`

## Order

1. Verify the environment and repository checkout.
2. Implement and test the trainable Gaussian MLP ensemble.
3. Rebuild one Hopper seed end-to-end.
4. Reproduce the calibration evaluator.
5. Select the baseline configuration using development data only.
6. Freeze the baseline.
7. Run a final baseline test evaluation.
8. Implement Dynamics / Observation / Policy interventions.
9. Implement CRN pairing and intervention audits.
10. Run multi-seed Hopper.
11. Replicate on Walker2d.
12. Rebuild thesis figures/tables from machine-readable outputs.

## Artifact policy

Keep source code, configs, metadata, tables, and final figures in Git. Keep large model checkpoints and raw downloaded datasets outside Git (for example Colab/Drive), referenced by metadata and manifests.

