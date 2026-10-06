# Placeholder checklist — Paper 1 drafts

Total placeholders: **38**


## REF (26)

- [ ] **02_introduction.md:3** — `[REF: Levine et al., offline RL tutorial; Kidambi et al., MOReL; Yu et al., MOPO; Janner et al., MBPO]`
      context: Model-based reinforcement learning relies on learned world models that predict the next state from the current state and action. When such a model is trained on
- [ ] **02_introduction.md:3** — `[REF: Lakshminarayanan et al., deep ensembles; Chua et al., PETS]`
      context: Model-based reinforcement learning relies on learned world models that predict the next state from the current state and action. When such a model is trained on
- [ ] **02_introduction.md:3** — `[REF: Kuleshov et al., calibrated regression]`
      context: Model-based reinforcement learning relies on learned world models that predict the next state from the current state and action. When such a model is trained on
- [ ] **02_introduction.md:3** — `[REF: Malik et al., calibrated model-based RL]`
      context: Model-based reinforcement learning relies on learned world models that predict the next state from the current state and action. When such a model is trained on
- [ ] **02_introduction.md:5** — `[REF: Ovadia et al., can you trust your model's uncertainty]`
      context: Calibration is known to degrade under distribution shift [REF: Ovadia et al., can you trust your model's uncertainty]. Most studies, however, treat shift as one
- [ ] **03_related_work.md:7** — `[REF: Lakshminarayanan et al. 2017]`
      context: Deep ensembles combine independently trained networks that each output a Gaussian mean and variance, and use the mixture or moment-matched aggregate as the pred
- [ ] **03_related_work.md:7** — `[REF: Chua et al., PETS]`
      context: Deep ensembles combine independently trained networks that each output a Gaussian mean and variance, and use the mixture or moment-matched aggregate as the pred
- [ ] **03_related_work.md:7** — `[REF: Fort et al., deep ensembles loss landscape]`
      context: Deep ensembles combine independently trained networks that each output a Gaussian mean and variance, and use the mixture or moment-matched aggregate as the pred
- [ ] **03_related_work.md:7** — `[REF: Ovadia et al. 2019]`
      context: Deep ensembles combine independently trained networks that each output a Gaussian mean and variance, and use the mixture or moment-matched aggregate as the pred
- [ ] **03_related_work.md:11** — `[REF: Kuleshov et al. 2018]`
      context: Classification calibration is commonly summarized with expected calibration error over confidence bins. For continuous outputs, the analogous notion is that a n
- [ ] **03_related_work.md:11** — `[REF: Levi et al., evaluating and calibrating uncertainty prediction in regression tasks]`
      context: Classification calibration is commonly summarized with expected calibration error over confidence bins. For continuous outputs, the analogous notion is that a n
- [ ] **03_related_work.md:11** — `[REF: Song et al., distribution calibration for regression]`
      context: Classification calibration is commonly summarized with expected calibration error over confidence bins. For continuous outputs, the analogous notion is that a n
- [ ] **03_related_work.md:11** — `[REF: Kuleshov & Deshpande 2022]`
      context: Classification calibration is commonly summarized with expected calibration error over confidence bins. For continuous outputs, the analogous notion is that a n
- [ ] **03_related_work.md:15** — `[REF: Levine et al. 2020]`
      context: Offline RL learns from fixed data, and the mismatch between the data distribution and the distribution induced by a learned policy is a central difficulty [REF:
- [ ] **03_related_work.md:15** — `[REF: Kidambi et al., MOReL; Yu et al., MOPO; Janner et al., MBPO]`
      context: Offline RL learns from fixed data, and the mismatch between the data distribution and the distribution induced by a learned policy is a central difficulty [REF:
- [ ] **03_related_work.md:15** — `[REF: Malik et al. 2019]`
      context: Offline RL learns from fixed data, and the mismatch between the data distribution and the distribution induced by a learned policy is a central difficulty [REF:
- [ ] **03_related_work.md:15** — `[REF: related work on shift taxonomies in RL; to be completed from literature scan]`
      context: Offline RL learns from fixed data, and the mismatch between the data distribution and the distribution induced by a learned policy is a central difficulty [REF:
- [ ] **03_related_work.md:15** — `[REF: to be verified in literature scan]`
      context: Offline RL learns from fixed data, and the mismatch between the data distribution and the distribution induced by a learned policy is a central difficulty [REF:
- [ ] **03_related_work.md:19** — `[REF: Pearl, Causality; Hernán & Robins, Causal Inference: What If]`
      context: We borrow the language of potential outcomes and interventions [REF: Pearl, Causality; Hernán & Robins, Causal Inference: What If]. In a simulator, the analyst 
- [ ] **03_related_work.md:19** — `[REF: Spirtes et al., Causation, Prediction, and Search]`
      context: We borrow the language of potential outcomes and interventions [REF: Pearl, Causality; Hernán & Robins, Causal Inference: What If]. In a simulator, the analyst 
- [ ] **03_related_work.md:23** — `[REF: Bottou et al., counterfactual reasoning and learning systems; Thomas & Brunskill, data-efficient off-policy evaluation; Jiang & Li, doubly robust off-policy value evaluation]`
      context: Counterfactual evaluation asks how a system would have performed under a different policy or condition. Off-policy evaluation estimates policy value from logged
- [ ] **03_related_work.md:27** — `[REF: Ovadia et al.; additional corruption-robustness and calibration studies]`
      context: Several studies report that calibration degrades with shift severity or that uncertainty estimates become unreliable on corrupted inputs [REF: Ovadia et al.; ad
- [ ] **03_related_work.md:27** — `[REF: recent horizon-calibration world-model papers; to be verified]`
      context: Several studies report that calibration degrades with shift severity or that uncertainty estimates become unreliable on corrupted inputs [REF: Ovadia et al.; ad
- [ ] **03_related_work.md:27** — `[REF: attribution of performance changes under distribution shift; to be verified]`
      context: Several studies report that calibration degrades with shift severity or that uncertainty estimates become unreliable on corrupted inputs [REF: Ovadia et al.; ad
- [ ] **04_method.md:71** — `[REF: Kuleshov et al. 2018]`
      context: We did not use the classification ECE. That metric bins predicted confidences of discrete labels. A continuous regression output has no analogous single confide
- [ ] **07_discussion.md:25** — `[REF: Yu et al., MOPO; Kidambi et al., MOReL]`
      context: Planners that use ensemble uncertainty as a penalty or as a trust signal assume that stated uncertainty tracks error [REF: Yu et al., MOPO; Kidambi et al., MORe

## FILL (7)

- [ ] **04_method.md:105** — `[FILL: bootstrap resampling unit, number of resamples.]`
      context: For hypothesis tests we used a paired sign-flip permutation test on the seed-level differences. For $n$ seeds, the exact test has $2^n$ equally likely sign assi
- [ ] **05_experimental_setup.md:12** — `[FILL: Minari version and dataset hash or commit.]`
      context: Dataset names, versions, episode boundaries, and metadata were recorded in a manifest committed with the code. [FILL: Minari version and dataset hash or commit.
- [ ] **05_experimental_setup.md:59** — `[FILL: device, GPU/CPU model, driver and library versions]`
      context: Training and evaluation were run on [FILL: device, GPU/CPU model, driver and library versions]. Random generators were seeded per training seed and per evaluati
- [ ] **05_experimental_setup.md:59** — `[FILL: determinism flags used, if any.]`
      context: Training and evaluation were run on [FILL: device, GPU/CPU model, driver and library versions]. Random generators were seeded per training seed and per evaluati
- [ ] **05_experimental_setup.md:59** — `[FILL: repository URL, commit hash, and archive DOI.]`
      context: Training and evaluation were run on [FILL: device, GPU/CPU model, driver and library versions]. Random generators were seeded per training seed and per evaluati
- [ ] **06_results.md:105** — `[FILL: sign-flip permutation $p$-values and Holm-adjusted $p$-values for each of the twelve conditions, from the results manifest. At $n = 10$ the minimum attainable two-sided $p$ is 0.00195; at $n = 5$ it is 0.0625, so no Walker2d condition can reach $p < 0.05$ in the exact test even before the Holm correction.]`
      context: [FILL: sign-flip permutation $p$-values and Holm-adjusted $p$-values for each of the twelve conditions, from the results manifest. At $n = 10$ the minimum attai
- [ ] **06_results.md:130** — `[FILL: secondary-metric ATE tables, if the verified numbers are available; otherwise state that they were not analyzed.]`
      context: We note what the negative dynamics effects do and do not show. The metric is an unsigned deviation from nominal coverage. An intervention that moves the error d

## VERIFY (5)

- [ ] **04_method.md:25** — `[VERIFY: confirm that the evaluated intervals use a Gaussian with this mean and total variance, rather than quantiles of the member mixture.]`
      context: Members were trained independently. The ensemble predictive distribution for a given input was summarized by a mean $\mu = \frac{1}{5}\sum_j \mu_j$ and a total 
- [ ] **04_method.md:31** — `[VERIFY: confirm which of test and probe is used for each of D, O, P in the released code; update this sentence accordingly.]`
      context: The roles are distinct. The train split fits the network weights and the standardization statistics. The calibration split is used only to fit the recalibration
- [ ] **04_method.md:43** — `[VERIFY: confirm that it is computed per dimension.]`
      context: where $f$ is the noise fraction, 0.05 (low) or 0.10 (high). Here `std` is the standard deviation of the test observations. [VERIFY: confirm that it is computed 
- [ ] **04_method.md:65** — `[VERIFY: confirm that the per-instance aggregation runs over output dimensions in this way.]`
      context: [VERIFY: confirm that the per-instance aggregation runs over output dimensions in this way.] The instance-level calibration error is
- [ ] **04_method.md:105** — `[VERIFY: confirm family definition against the analysis code.]`
      context: For hypothesis tests we used a paired sign-flip permutation test on the seed-level differences. For $n$ seeds, the exact test has $2^n$ equally likely sign assi
