# Prospective transformation-control replication

This protocol fixes the experiment after reviewing the new controls and before generating its fresh data. The [configuration](../configs/transformation_replication.json) evaluates a training procedure on one synthetic generator family. It does not establish architecture superiority, quantum advantage or a minimum-description-length result. Existing manuscript results and the implementation smoke remain separate evidence.

## Primary question and analysis unit

Does the original quantum encoder, restricted to two retained Z expectations and an affine head, have a different shifted-test forecast error from the original Givens encoder with the same two-real-coordinate interface and head?

The primary contrast is `qte_shared - cte_shared` in native next-step forecast MSE. Positive differences favor the classical shared-head model. First average the four independent test sequences within each shifted condition, then equally average the five conditions, then equally average all three initialization seeds. Report the eight paired data-realization differences, their mean and observed range. Also report every initialization-specific difference. No best initialization is selected. The eight data realizations are the analysis units; initialization seeds, sequences, timesteps and conditions are not additional independent realizations. No significance threshold, power claim or test-driven stopping rule is specified.

In-distribution forecasting, each separate shift, original `qte` versus `cte`, gains/offsets, phase and entanglement ablations, analytic references, trained/untrained comparisons and common reconstruction/forecast probes are secondary. Coordinatewise permutation entropy, LZ76 and lag-one correlation are exploratory. Descriptor distance is not an MDL codelength and cannot establish information preservation. All named models and conditions must be reported regardless of which performs best.

## Fixed data

Data seeds are 223, 227, 229, 233, 239, 241, 251 and 257. Each seed defines an independent orthogonal observation mixing matrix and separately seeded training, validation and test sequences. No observations from these seeds were used to choose this configuration. Earlier diagnostic, regression and forecasting seeds are excluded explicitly in the configuration. Initialization seeds are 401, 409 and 419 and are crossed with every data realization.

Each sequence has 128 observations, four bounded observed features and two generating coordinates. Each realization has four training and four validation sequences, plus four test sequences in each of six conditions. The generator remains `pilot.data`: three consecutive piecewise Gaussian AR regimes, orthogonal observation mixing, Gaussian measurement noise and `tanh` observations. The test conditions are in-distribution, mean shift, standard-deviation shift, persistence shift, observation-noise shift and their combination. The historical field `variance_ratio` multiplies standard deviation; middle-regime variance multipliers are 2.25 in the base profile and 9 under its shift. Innovation standard deviation scales by `sqrt(1-rho**2)` so persistence changes preserve within-regime stationary variance.

Conditions use distinct random innovations rather than matched counterfactual sequences. Thus a finite-sample difference between two conditions also includes sampling variation. A changepoint transition is scored like every other transition; regime timing is fixed and equally shared by models. Forecasting is one-step and teacher-forced. GRU inputs stop at the current observation and sequence memory is reset between sequences. This study has no recursive multi-step endpoint. Models receive observed features only; generating states are saved for provenance and never supplied to fitting, selection or probes.

## Models and comparison limits

| Models | Controlled comparison | Limit |
| --- | --- | --- |
| `cte_shared`, `qte_shared` | Two real readouts, identical initial affine heads, 16 trained parameters each and the same forward-difference encoder/analytic-head Adam procedure | Encoder function classes, feature ranges and loss surfaces differ |
| `cte`, `qte`, `qte_noent` | Original native Givens and quantum constructions | Two retained qubits do not equal two classical real coordinates in state capacity |
| `cte_affine` | Original Givens encoder/decoder plus two gains and four offsets; reproduces original outputs at initialization | One ring block is still a restricted orthogonal family |
| `qte_ry`, `qte_ry_noent` | RY-only circuits at the original tied angle count, with and without CZ | RY-only changes the circuit and optimization geometry; CZ can still entangle real amplitudes |
| `encoding_ridge`, `encoding_reduced_rank` | Exact 81-column bounded-RY product feature span, with full or rank-two linear forecasting coefficients | Analytic coefficient capacity and regularization are not matched to the circuit |
| `reduced_rank`, `mlp_te`, `gru_te`, `pca`, `random_linear`, `persistence` | Linear, nonlinear, recurrent, untrained projection and full-input forecasting references | Native PCA/random-projection objectives are reconstruction; forecast probes are separate |

Every gradient-trained model has an untrained case with the same seeded architecture and initialization. Analytic fits do not receive a misleading untrained label. Persistence retains all four observed features. `encoding_ridge` retains all 81 basis coordinates. Both are labeled uncompressed. The constant feature column is removed by training-only centering and represented by the unpenalized output intercept; the declared 81 coordinates are an upper-bound basis size, not 81 independent fitted directions.

The original Givens code remains unchanged. Its nonrecurrent native map has two unit nonzero singular values; decoder-input gains remove that restriction and output offsets remove the zero-intercept restriction. The original quantum decoder receives a reduced density matrix, not merely two Z expectations. Compression uses a trace-preserving discard-and-replace channel, `Tr_trash(rho) tensor |0><0|_trash`. It preserves the retained marginal and its mixedness but removes correlations with the discarded subsystem. It neither conditions on a zero measurement outcome nor preserves entanglement with a removed subsystem. In Qiskit's qubit ordering, the replacement trash state is the left tensor factor. The shared quantum head accesses only its two retained Z expectations.

The explicit feature derivation applies to one data embedding followed by data-independent channels and linear expectation readouts. It does not extend automatically to input reuploading or recurrent quantum models. The replication therefore contains nonrecurrent quantum/Givens models. The GRU is a separate recurrent classical reference.

## Optimization, selection and diagnostics

Every gradient candidate completes 64 epochs at each of four learning rates: 0.001, 0.005, 0.02 and 0.08. Each candidate starts from the same seeded initialization. Native validation MSE chooses the epoch and learning rate; ties keep the first epoch/candidate. Initial epoch-zero predictions remain available through the untrained control but are not native checkpoint candidates. Native analytic forecasting and common probe ridge penalties are 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1 and 1.0. Fits and centering use training only. Common probes do not select native checkpoints.

Quantum native training uses forward differences with width 1e-5 and exact density matrices. Original Givens, its affine extension, MLP and GRU use autograd. The primary shared pair uses the same finite-difference encoder gradients, analytic head gradients and Adam implementation. The comparison matches candidate and epoch budgets, not wall time or overall expressive capacity. Native quantum decoders use the existing data-free feature-neutral calibration with seeded jitter; shared heads replace the native decoder and use near-zero encoders plus identical seeded affine heads. Calibration is checked separately for every initialization and accesses only a zero reference observation.

For every gradient candidate, compare median training MSE in epochs 33-48 with epochs 49-64. Flag a relative decrease exceeding 1%. Also flag a best validation epoch in 49-64 and report whether the selected learning rate or ridge penalty lies on a grid boundary. These diagnostics never change fitting, budgets, selected endpoints or test inclusion. Continuing improvement or boundary selection exposes a limitation of the fixed search. Absence of a flag does not establish global convergence. A follow-up prompted by these diagnostics must fix another protocol before accessing another untouched test sample.

## Lock, replay and execution

The mandatory data-free preflight checks configuration, pinned core dependencies, head initialization and native quantum calibration without generating observations or fitting models. All 624 trained/untrained model and probe cases across the 24 data/initialization groups are fitted, saved and replayed on validation before any test data are generated. A replay mismatch stops the run. Source commit and consumed source/protocol/dependency hashes are checked again after fitting. The configuration, datasets, selections, constant predictions, checkpoints, probes and replay evidence are hash-locked. The lock file's own digest is saved in the manifest before test access.

Only then are independent test partitions generated. Their training/validation sections must reproduce exactly. Test evaluation uses restored checkpoints and frozen saved probe coefficients. No fitting or selection occurs after lock. The manifest records fitting, locked, evaluating and complete phases so interrupted output cannot be accepted as a completed run. A failed run stops; an existing output directory is never overwritten or silently resumed.

The planned output is 29,952 sequence/task rows and 384 constant-reference rows. There are 61,440 gradient updates in total. Four isolated CPU fitting processes use one Torch thread each. Set BLAS/OpenMP threads to one as shown below. There is no GPU, hardware quantum execution or measurement-shot noise. This is a substantially longer run than the implementation smoke; no runtime is inferred from tests on the fresh seeds.

The dedicated report verifies complete unique coverage, every candidate budget and validation selection, probe metadata, all hashes and regenerated data. It independently restores every saved model and replays every native and probe test score without refitting. Constant references are recalculated from training-only means. A discrepancy or nonfinite result blocks the report. Full outcomes, all optimization diagnostics, per-realization and per-initialization contrasts and the complete archive are retained. Existing reports are never overwritten by this report path.

From a clean committed checkout with the pinned CPU dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.preflight --config configs/transformation_replication.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.fresh_study --config configs/transformation_replication.json --output pilot_runs/transformation_replication
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.replication_report --input pilot_runs/transformation_replication --output docs/transformation_replication
```

The full run has not started during final review. None of its fresh training, validation or test observations have been generated.

## Scope and supporting sources

Fresh seeds protect this replication from the previously inspected outcomes; they do not make its design independent of earlier development or provide external validity. The primary endpoint is a new prospectively specified comparison informed by earlier exploratory work. Eight realizations of one known generator family remain a narrow benchmark. A substantially different nonlinear generator or real dataset would be needed for broader generalization claims. Coordinatewise descriptors and fixed correlation/entropy summaries do not test description-length estimation.

- [IBM Qiskit DensityMatrix documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.DensityMatrix): partial reset evolves by the reset channel, preserving the density-matrix description rather than sampling a pure postmeasurement state.
- [Schuld (2021), Supervised quantum machine learning models are kernel methods](https://arxiv.org/abs/2101.11020): supports analysis of the data-encoding feature space. The repository's exact bounded-RY product basis follows from its embedding directly.
- [Bowles, Ahmed and Schuld (2024), Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059): motivates strong classical references and separating representation changes from training and selection choices.
