# Paired initialization and training-budget control

This follow-up uses training and validation partitions only. Test partitions are not generated. It diagnoses initialization and optimization; it is not an architecture ranking or a test-set study.

## Prespecified comparison

The quantum autoencoder and quantum transition encoder each receive `near_zero` and `feature_neutral` initialization. Classical ring reconstruction/forecast models, MLP reconstruction, GRU forecasting, PCA and reduced-rank forecasting provide validation references. Two independent data realizations use seeds 17 and 41, with initialization seed 101. Sequences have length 64, with two training and two validation sequences per realization. Every trained case completes 16 epochs at learning rate 0.02. Report the initial error and minimum validation error within the first 8 and all 16 epochs from the same trajectory. These are nested budget summaries, not independent repetitions.

The initialization control changes only the existing first decoder block's discarded-qubit rotation parameters. The gates, parameter tying, parameter count, encoder parameters and remaining decoder parameters stay the same. The runner verifies that paired initial encoder readouts have identical SHA-256 hashes.

For `feature_neutral`, bounded nonlinear least squares finds decoder-parameter centers whose discarded-coordinate readout is zero on a synthetic zero-observation input. This reference is constructed without accessing any generated sequence. The center residual must be below 1e-8. The original seeded parameter jitter is added to those centers to avoid an exact symmetry point; actual jittered readouts need not be exactly zero. Calibration evaluations, centers, residuals and timing are recorded. Default `near_zero` behavior remains unchanged.

## Interpretation

A substantial validation improvement from a decoder-only initialization change would identify an optimization confound in comparisons made with a short near-zero training budget. It would not prove that the quantum architecture is superior or that its capacity equals a classical model's capacity. Native decoder scores and common ridge-probe validation scores remain separate.

Probe coefficients are fit on training latents and penalties are selected on validation. No test score enters this control. The short sequences are suitable for optimization diagnostics, not stable multiscale complexity conclusions. The existing two-epoch and eight-epoch reports remain historical diagnostic records with their original source manifests.

## Run

```bash
python -m pilot.validation_control --config configs/initialization_control.json --output pilot_runs/initialization_control
```

The configuration remains exploratory: two data realizations, one initialization and one learning rate do not support calibrated confidence intervals. If calibration fails for another circuit configuration, the initializer raises an error rather than silently accepting a non-neutral center. Defaults are preserved until the paired control is reviewed.

Calibration uses [SciPy's bounded nonlinear least-squares solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html). It changes initial values, not the training objective.
