Understanding of Figure 7
- Figure 7 studies how estimation accuracy for heterogeneous treatment effects evolves as the first-stage DNN trains. The x-axis is the DNN training epoch; the left y-axis is the DNN training MSE, and the right y-axis is the mean absolute percentage error (MAPE) of estimated treatment effects.
- The DeDL (debiased deep learner) and SDL (single deep learner) estimators both rely on the DNN to learn nuisance functions. When the DNN is poorly trained (roughly the first ≲100 epochs), both have high MAPE that is worse than the simpler linear regression (LR) baseline—debiasing does not help if the nuisance model is weak.
- As the DNN converges (training MSE shrinks), DeDL begins to outperform both SDL and LR. The debiasing step becomes effective once the nuisance model is sufficiently accurate, leading to noticeably lower MAPE for DeDL in later epochs.
- Practical takeaway from the paper: monitor DNN training error as a proxy for nuisance-quality; once it is low, debiased estimators gain a clear advantage and deliver more reliable causal effect estimates.

Roadmap for synthetic replication of Figure 7
- Goal: recreate the Figure 7 pattern with synthetic data—training MSE decreases over epochs while DeDL’s estimation MAPE eventually beats SDL and LR as the DNN is trained longer.
- Data generation: simulate m binary treatments (combinatorial assignments) and d covariates. Outcomes come from a nonlinear, interaction-heavy function of (X, T), ensuring LR is misspecified while a well-trained DNN can capture the structure. Noise is added to keep the task realistic.
- True effects: compute ground-truth average treatment effects (ATEs) for every treatment combination via the known data-generating function over a large Monte Carlo sample.
- Estimators: (1) LR: OLS on (X, T) without higher-order interactions; (2) SDL: plug-in ATEs from a DNN trained on (X, T); (3) DeDL: doubly robust ATEs using the same DNN outcome model plus propensity scores (known from random assignment) to orthogonalize residuals.
- Training loop: train the DNN for ~400 epochs, log training MSE each epoch, and at checkpoints compute MAPEs for LR (fixed), SDL, and DeDL against true ATEs.
- Success criterion (to iterate toward): early epochs show DeDL≈SDL and both worse than LR; as epochs and validation/training error drop, DeDL’s MAPE dips below SDL and LR.

Replication summary and findings
- Data generation implemented in `Replication/Figure7.py`: m=3 binary treatments, d=10 covariates, nonlinear sigmoid + interaction outcome with additive noise; random assignment gives known propensities. True ATEs computed via noise-free Monte Carlo over evaluation X.
- Estimators: LR (OLS on X,T), SDL (DNN plug-in), DeDL (doubly robust with the same DNN plus propensity weighting). DNN: 2 hidden layers (64 units), Adam lr=1e-3, 400 epochs, batch 128.
- Resulting plot: `Replication/figure7_synthetic.png` shows training MSE decays quickly; DeDL MAPE starts comparable to SDL and better than LR after few epochs; as training continues and MSE shrinks, both DeDL and SDL converge to very low MAPE, with DeDL slightly below SDL and both well below LR.
- Alignment with paper’s Figure 7: qualitatively matches the pattern that as the DNN trains, debiased estimators improve and beat LR; early-epoch gap between DeDL and SDL is modest in this synthetic setup (noise and moderate misspecification). LR stays flat and worse than the trained DNN-based estimators.
