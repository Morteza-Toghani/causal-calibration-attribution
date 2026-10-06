# 3. Related Work

We organize prior work into six areas and then state how this study differs. Citations are placeholders to be resolved before submission, and the literature scan should be refreshed before the final version.

## 3.1 Deep ensembles and predictive uncertainty

Deep ensembles combine independently trained networks that each output a Gaussian mean and variance, and use the mixture or moment-matched aggregate as the predictive distribution [REF: Lakshminarayanan et al. 2017]. They are simple, parallel, and require little tuning, and they became a standard baseline for predictive uncertainty and for probabilistic dynamics models [REF: Chua et al., PETS]. Fort et al. interpret their benefit through the diversity of modes reached by different initializations [REF: Fort et al., deep ensembles loss landscape]. Ovadia et al. evaluated several uncertainty methods under dataset shift and found that ensembles were among the more robust methods, while all methods degraded as shift grew [REF: Ovadia et al. 2019]. That evaluation concerns classification benchmarks and treats shift as a single severity axis. Our work uses the same model family in a regression world-model setting, and separates shift by mechanism.

## 3.2 Calibration for regression

Classification calibration is commonly summarized with expected calibration error over confidence bins. For continuous outputs, the analogous notion is that a nominal-p predictive interval should contain the target with frequency p. Kuleshov et al. formalized this for regression, proposed a recalibration procedure, and evaluated it in model-based RL [REF: Kuleshov et al. 2018]. Levi et al. studied calibrated regression through variance rescaling [REF: Levi et al., evaluating and calibrating uncertainty prediction in regression tasks], and Song et al. proposed a local, distribution-level calibration notion [REF: Song et al., distribution calibration for regression]. Kuleshov and Deshpande later connected calibration with sharpness [REF: Kuleshov & Deshpande 2022]. Our recalibration step is a single per-seed variance scaling, closest in form to the variance-rescaling approach; we use it as an analysis device for a controlled comparison rather than as a proposed method. Our primary metric is a regression-specific, instance-level variant, and we do not use classification ECE.

## 3.3 Distribution shift in offline RL and world models

Offline RL learns from fixed data, and the mismatch between the data distribution and the distribution induced by a learned policy is a central difficulty [REF: Levine et al. 2020]. Model-based offline methods use ensemble disagreement or predicted variance as a penalty on uncertain regions [REF: Kidambi et al., MOReL; Yu et al., MOPO; Janner et al., MBPO]. These methods assume the uncertainty is informative where the model is wrong. Malik et al. showed that calibrated uncertainty improved model-based RL [REF: Malik et al. 2019]. Work on distribution shift in RL has also separated shift sources, for example covariate, dynamics, and observation shift, and studied their effect on performance or detection [REF: related work on shift taxonomies in RL; to be completed from literature scan]. The question of how each source affects the calibration of a fixed trained world model, with an effect estimate per mechanism, has received less direct treatment [REF: to be verified in literature scan].

## 3.4 Causal inference with interventions

We borrow the language of potential outcomes and interventions [REF: Pearl, Causality; Hernán & Robins, Causal Inference: What If]. In a simulator, the analyst can set the treatment directly and keep other factors fixed, which makes the paired difference between a treated and an untreated evaluation a well-defined effect under the experimental design. The structural-causal tradition also provides tools for discovering causal structure from observational data [REF: Spirtes et al., Causation, Prediction, and Search]. We do not use structure discovery. Our identification relies on the experiment itself, and we do not claim identification outside the simulator.

## 3.5 Counterfactual evaluation in RL

Counterfactual evaluation asks how a system would have performed under a different policy or condition. Off-policy evaluation estimates policy value from logged data using importance weighting, doubly robust estimators, or learned models [REF: Bottou et al., counterfactual reasoning and learning systems; Thomas & Brunskill, data-efficient off-policy evaluation; Jiang & Li, doubly robust off-policy value evaluation]. Those methods estimate returns. Our outcome is the calibration of a one-step predictive distribution, and our counterfactual is constructed by re-evaluating a fixed model under a controlled intervention, not by reweighting logged data. The shared idea is the paired comparison between a factual and a counterfactual condition.

## 3.6 Studies of miscalibration under shift

Several studies report that calibration degrades with shift severity or that uncertainty estimates become unreliable on corrupted inputs [REF: Ovadia et al.; additional corruption-robustness and calibration studies]. Recent work on world models examines calibrated uncertainty over long horizons [REF: recent horizon-calibration world-model papers; to be verified]. Studies of failure attribution under shift in supervised learning ask which factor drives a performance drop [REF: attribution of performance changes under distribution shift; to be verified]. These works motivate our question but, to our knowledge from the scan so far, they do not estimate per-mechanism calibration effects for a probabilistic offline world model with paired interventions. This statement should be re-checked against the literature before submission.

## 3.7 Positioning of this work

The contribution is attribution, not detection or mitigation. Detection asks whether calibration is failing. Mitigation asks how to repair it. We ask which mechanism produces the failure and how large the effect is under a controlled intervention, and we report how the answer depends on two analysis choices (re-simulation baseline and baseline recalibration) that earlier evaluations often leave implicit. Table 3.1 summarizes the positioning; entries to be completed during the literature scan.

| Line of work | Mechanisms separated | Calibration measured | Paired counterfactual effect |
|---|---|---|---|
| Deep ensembles [REF] | no | partly | no |
| Regression calibration [REF] | no | yes | no |
| Calibrated model-based RL [REF] | no | yes | no |
| Shift robustness benchmarks [REF] | partly | yes | no |
| Off-policy evaluation [REF] | policy | no (value) | not for calibration |
| This work | dynamics, observation, policy | yes | yes |
