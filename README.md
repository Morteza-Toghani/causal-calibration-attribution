on

The primary analysis uses 1-step transition prediction.

Multi-step evaluation is treated as an extension because rollout compounding and policy feedback can introduce additional sources of variation into attribution.

---

Calibration and Evaluation

The primary outcome is a regression-appropriate measure of calibration error.

Secondary metrics include:

- Predictive negative log-likelihood (NLL)
- Prediction-interval coverage
- Sharpness
- CRPS
- Predictive uncertainty
- Ensemble disagreement

The project distinguishes calibration from sharpness and does not treat a single metric as sufficient evidence of reliable uncertainty estimation.

---

Statistical Validation

The analysis is designed around matched observations and repeated model instances.

Planned statistical components include:

- Paired differences
- Multi-seed evaluation
- Bootstrap confidence intervals
- Paired permutation tests
- Effect sizes
- Holm correction for multiple comparisons

The analysis will report uncertainty around effect estimates rather than relying only on statistical significance.

---

Evidence Status

IN DESIGN

The experimental protocol, repository architecture, and core methodological design are being implemented and tested.

No final empirical claim is made here before the corresponding experiment, statistical analysis, and reproducibility checks are complete.

Evidence status will be updated as the project progresses:

"IN DESIGN → PROTOTYPE → PRELIMINARY → VALIDATED → REPRODUCIBLE"

---

Results

No final results are reported yet.

Once experiments are complete, this section will contain only results that can be traced to:

Research Question → Hypothesis → Intervention → Configuration → Experiment → Statistical Analysis → Result

Unsupported or placeholder numbers will not be included.

---

Limitations

The main limitations currently considered include:

- controlled simulation does not establish causal effects in real-world deployment;
- causal interpretation depends on the intervention design and its assumptions;
- policy intervention in the primary analysis is evaluated on fixed probe states rather than unrestricted rollout;
- calibration behaviour may depend on model architecture, environment, dataset, and random seed;
- multi-step uncertainty propagation is outside the primary thesis-level outcome.

These limitations will be updated as the experimental evidence develops.

---

Reproducibility

The repository is designed so that experiments can be reconstructed from explicit configurations rather than undocumented manual steps.

Each experiment should record:

- environment
- dataset version / manifest
- model configuration
- random seed
- intervention mechanism
- intervention severity
- output location

Large datasets remain outside Git; only manifests and reproducibility metadata belong in the repository.

---

Repository Structure

causal-calibration-world-models/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── model/
│   ├── shift/
│   └── experiment/
│
├── data/
│   └── manifests/
│
├── src/
│   ├── envs/
│   ├── models/
│   ├── shifts/
│   ├── metrics/
│   ├── causal/
│   └── stats/
│
├── experiments/
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── notebooks/
├── docs/
└── tests/

The "causal/" layer is explicit because causal attribution is a core methodological component of the thesis rather than a secondary analysis hidden inside notebooks.

The "docs/" directory will document the counterfactual design, causal assumptions, methodology, limitations, and related work.

---

Research Scope

This repository corresponds primarily to the MSc thesis / Paper 1 diagnosis stage of a broader research program:

Thesis → Paper 1 → Paper 2 → PhD

The thesis establishes the mechanism-specific calibration-failure and attribution framework.

Later work may extend the diagnosis toward causally-informed recalibration and safety-constrained decision-making, but those components are not treated as completed contributions in this repository.

---

Author
Morteza Toghani
MSc Artificial Intelligence & Robotics
