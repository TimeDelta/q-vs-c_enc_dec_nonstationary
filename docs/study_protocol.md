# Protocol 2: publication development plan

## Question and scope

Does a compressed representation's temporal structure predict reconstruction and one-step forecasting performance under controlled distribution shifts after accounting for validation distortion, model family and training budget?

Complexity matching is a hypothesis to test, not a necessary condition for useful representations. Representation descriptors depend on coordinates, quantization, sample length and readout. A compensated invertible transformation can preserve predictions while changing some descriptors. The pipeline records scale, orthogonal rotation and time-shuffle controls for this reason.

The quantum branch is an exact classical simulation of small density matrices. The comparison assesses these model constructions under this protocol. It does not establish hardware speedup, quantum advantage or equal information capacity between k real coordinates and k qubits.

## Data and partitions

Observations are bounded nonlinear mixtures of a low-dimensional Gaussian autoregressive process. Three segments change mean, variance and persistence. Innovation variance is adjusted by sqrt(1-rho^2), keeping stationary variance independently controllable from persistence. Boundaries still have transients. A tanh observation function gives a common feature domain [-1, 1] without fitting a test-set scaler.

Train, validation and test sequences use separate deterministic RNG streams. Test regimes are in-distribution, increased mean shift, variance shift, persistence shift, observation noise shift and a combined shift. Source latent trajectories are saved, but are not supplied to model training or selection. All files have split hashes and regime metadata.

Forecasting is teacher-forced one-step prediction, not autonomous rollout or change-point detection. Both tasks use the same T-1 input prefix to align sample counts. The bounded observation domain may mask extreme changes; later experiments should include additional observation functions and real datasets with leakage-safe temporal splits.

## Models and compression

Eight original quantum/classical, reconstruction/forecast and recurrent/feedforward combinations are corrected. Two quantum models without entangling gates test circuit structure. Classical MLP and GRU models, PCA, reduced-rank regression and random projection provide alternatives. Raw persistence has no compression and is labelled an uncompressed reference.

Classical ring models zero fixed discarded coordinates before decoding. Quantum models apply a trace-preserving discard/reset channel to fixed discarded qubits before decoding. No data-dependent subsystem selection is allowed. Recurrence retains only that compressed object. Quantum mixing keeps density matrices intact.

Quantum inputs use RY(arccos(x)) for bounded observations and outputs use exact Z expectations. Temporal complexity is measured on the retained single-qubit Z expectations, a restricted classical readout of the quantum representation. It is not an informationally complete state reconstruction. Common probes therefore compare accessible readouts, not the full retained quantum state.

## Selection and evaluation

Every learning-rate candidate completes its configured epoch budget. Epoch checkpoints and learning rates minimize validation native-task MSE. Test outcomes cannot enter the selection function. Analytic reduced-rank ridge penalties are also chosen on validation data.

For each selected representation, independent reconstruction and forecast ridge probes are fit on training latents and training targets. Ridge penalties are selected on validation only. Frozen probes are evaluated on each test regime. Native task scores and common probe scores are kept separate.

Each output row corresponds to a sequence, model and probe task. It includes native/probe MSE, readout descriptors and transformation checks. Native MSE is the architecture's native task score and repeats across the two probe-task rows; do not double-count it. `parameter_count` counts gradient-optimized scalar parameters; zero for analytic models does not imply zero fitted coefficients. Timing, gradient norms, loss evaluations and quantum input preparations are recorded. Common epoch budgets do not imply equal compute or equally optimized models.

Descriptors include normalized permutation entropy at scales 1, 2 and 4, median-threshold Lempel-Ziv phrase complexity and lag-one correlation. Descriptor mismatch is exploratory and has no universal zero target across source features and latent coordinates. It is not a minimum description length estimate. Short sequences can give unstable descriptor estimates.

## Verification and historical corrections

Regression checks cover objective dispatch, loss order independence, timestep weighting, finite input validation, quantum encoding/readout, valid mixed-state recurrence, genuine discard/reset compression, classical autograd, checkpoint compatibility, entropy label invariance and Hyperband budgets. Pilot checks cover data separation, probe selection, compensated rotation invariance, reduced rank and output manifests.

The original manuscript's loss-history derivatives describe optimization trajectories, not curvature in parameter space. Flat loss histories cannot diagnose barren plateaus. The new pilot records gradient norms but does not make a barren-plateau claim. Mixed states are not passed to pure-state Meyer-Wallach entanglement formulas.

## Work required before submission

1. Run the replicated pilot and inspect optimization failures, descriptor stability and runtime. Freeze any revised protocol before confirmatory data generation.
2. Increase independent data realizations and initialization repeats based on pilot variability and computational feasibility. Treat data realization as the main independent unit. Initialization repeats and multiple sequences from one realization are nested observations.
3. Predefine primary comparisons and shift regimes. Report paired effects and cluster-aware uncertainty rather than treating every row as independent. Adjust or clearly label multiple exploratory comparisons.
4. Test whether descriptor mismatch predicts test distortion beyond validation distortion with held-out data-seed evaluation. Avoid selecting descriptors or regression forms on confirmatory test outcomes.
5. Extend bottleneck sizes, sequence lengths, optimization budgets and observation mappings. Add shot/noise sensitivity only if hardware relevance is claimed. Compare fairly tuned strong classical baselines.
6. Write a results-driven manuscript with effect sizes, confidence intervals, failures, resource costs and readout limitations. Release all configurations and reproducibility manifests. Do not reuse original numerical conclusions without rerunning them.

## Relevant primary sources

- Bowles, Ahmed and Schuld (2024), [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059). Motivates strong classical baselines and circuit ablations; it does not determine the outcome of this time-series study.
- Hu et al. (2023), [Complexity Matters: Rethinking the Latent Space for Generative Modeling](https://arxiv.org/abs/2307.08283). Its generator-complexity results do not establish that temporal entropy must match between observations and representations.
- McClean et al. (2018), [Barren plateaus in quantum neural network training landscapes](https://doi.org/10.1038/s41467-018-07090-4). Parameter-gradient scaling must be tested directly before making this diagnosis.
- [EntroTS (2026)](https://doi.org/10.1016/j.chaos.2026.119042), an entropy-guided time-series representation study. Review the full paper before claiming novelty for entropy-based representation analysis.
