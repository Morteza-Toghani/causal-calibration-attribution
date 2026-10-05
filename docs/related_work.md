# Related work (working list; literature scan must be refreshed before any submission)

- Lakshminarayanan, Pritzel, Blundell (2017). Simple and Scalable Predictive Uncertainty Estimation using
  Deep Ensembles. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf
- Kuleshov, Fenner, Ermon (2018). Accurate Uncertainties for Deep Learning Using Calibrated Regression. ICML / PMLR 80.
  https://proceedings.mlr.press/v80/kuleshov18a.html
- Malik et al. (2019). Calibrated Model-Based Deep Reinforcement Learning. ICML / PMLR 97.
  https://proceedings.mlr.press/v97/malik19a.html
- Kuleshov & Deshpande (2022). Calibrated and Sharp Uncertainties in Deep Learning via Density Estimation.
  ICML / PMLR 162. https://proceedings.mlr.press/v162/kuleshov22a.html

## Positioning (as stated in the project plan)
Separating policy / observation / dynamics shift in RL has been studied before, so the separation itself is
not claimed as novel. The intended contribution is measuring the effect of each mechanism on calibration
failure of the predictive uncertainty of an offline world model with controlled, counterfactual-style
interventions. This repository makes no claim that no similar work exists. A systematic literature
positioning matrix is still to be written.
