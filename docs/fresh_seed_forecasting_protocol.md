# Prespecified fresh-seed forecasting replication

This protocol follows the initialization sensitivity diagnostic. Its scope is one-step, teacher-forced forecasting under controlled distribution shifts. It tests a bounded training procedure, not fully converged architectures, quantum advantage or neuroimaging generalization. The protocol and executable source are committed before fresh study fitting begins.

## Data and models

Eight independent generator realizations use data seeds 113, 127, 139, 151, 163, 179, 191 and 211. These seeds have not supplied the earlier exploratory reports. Each realization has its own observation mixing matrix and independently seeded sequences: two training, two validation and four test sequences per test condition, each of length 128. Input dimension is four, generating-state dimension is two and the enforced bottleneck is two. Three piecewise regimes occur within each sequence.

Initialization seed 303 is fixed for the replication. Variability across initialization seeds was checked in the earlier diagnostic; this study spends its bounded computation budget on independent data realizations. Its results condition on this initialization seed and cannot estimate optimization-seed variability.

Quantum transition encoders with circular CZ entanglement and without entanglement use feature-neutral initialization. Classical transition ring, MLP forecasting and GRU forecasting supply trained references. PCA, reduced-rank regression, random linear projection and persistence supply analytic references. Each trainable architecture also has an untrained control with identical seeded architecture and initialization. Zero-output and training-target-mean forecasts supply constant references.

## Selection and test separation

Every gradient-trained learning-rate candidate completes eight epochs, at rates 0.02 and 0.08. The candidate and epoch with lowest native validation MSE are selected. Ties retain the first candidate/epoch. Reduced-rank native ridge candidates are 0.001 and 0.01. Identical selection rules apply to classical and quantum models; this matches epoch budgets, not computation or capacity. Quantum gradients use forward differences of width 1e-5 and exact density-matrix simulation without sampling noise. Classical gradients use automatic differentiation.

Common linear probes fit coefficients on training latents and choose ridge penalty from 0.0001, 0.001 and 0.01 on validation. These probes are secondary representation measurements. They do not choose the native model checkpoint.

Independent data realizations may fit concurrently in four isolated processes, each with one Torch thread. This execution setting changes wall time, not the selection budget. All models and probe coefficients across all data seeds are fitted and saved before any test partitions are generated. A selection lock records file hashes of checkpoints, probe coefficients, training/validation data, configuration and selection diagnostics. Locked files must retain their hashes after evaluation. Test generation must reproduce the training/validation manifests. No model, hyperparameter, endpoint or analysis choice changes after test evaluation begins.

## Endpoints and analysis units

The primary endpoint is native forecast MSE, averaged equally across five shifted test conditions: mean shift, variance shift, persistence shift, noise shift and combined shift. Within a condition, average the four test sequences. The primary contrast is quantum transition encoder minus reduced-rank regression at this endpoint. Positive differences favor reduced-rank regression. Report all eight per-data-seed differences, their mean and observed range; no significance threshold or power claim is specified.

In-distribution native forecasting, the no-entanglement contrast, classical references, trained/untrained contrasts, common reconstruction/forecast probe MSE and each individual shift are secondary. Report all named references regardless of which performs best. Repeated sequences and shift conditions on a shared generator realization are not independent data units. Native reconstruction scores from PCA/random projection are not forecast endpoints.

Coordinate-wise permutation entropy, binary Lempel-Ziv complexity and lag-one correlation are exploratory descriptors. Scaling, compensated rotation and time-shuffling controls remain reported. Descriptor similarity is neither an MDL codelength nor proof of information preservation. At length 128, coarse-graining scales are included only when the descriptor sample threshold is met. The generator field `variance_ratio` multiplies standard deviation, so base/shift middle-regime variance multipliers are 2.25 and 9.

Eight independent realizations do not establish robustness beyond this generator family. The synthetic bounded observations are not clinical or neuroimaging data. Future model changes prompted by the results require another fresh evaluation.

## Run and reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.fresh_study --config configs/fresh_seed_forecasting.json --output pilot_runs/fresh_seed_forecasting
python scripts/report_fresh_seed_forecasting.py --input pilot_runs/fresh_seed_forecasting --output docs/fresh_seed_forecasting
```

The report includes the fixed configuration, environment/source manifest, selections, lock hashes, sequence-level outcomes, constant references, per-realization primary contrasts and a checkpoint/data archive. Runtime and model parameter counts are reported separately from predictive scores.
