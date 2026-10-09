# Temporal complexity and generalization under enforced compression

A reproducible research pipeline comparing quantum and classical encoder/decoder models on synthetic nonstationary time series. The corrected study asks whether temporal descriptors of compressed representations predict performance under distribution shifts beyond validation distortion.

**Status: protocol 2 implemented and smoke-tested. Publication claims require replicated experiments and further analysis.** The [historical manuscript](docs/historical_manuscript.md) records the original study but its results are superseded by corrections to task routing, scoring, compression and quantum state handling.

## Run the corrected experiment

Tested with Python 3.12 on CPU. Create a fresh virtual environment, then install:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-pilot.txt
python -m pilot.run --config configs/cpu_smoke.json --output pilot_runs/smoke
```

The [checked-in smoke evidence](docs/pilot_smoke/README.md) documents three exactly matching runs. The smoke configuration includes 17 architectures/baselines and 13 untrained controls, two training epochs and six held-out test regimes. It validates execution and finite outputs; it cannot establish architecture rankings, complexity matching or quantum advantage. Use a new output directory for each run. `--no-quantum` provides a faster classical-only check.

The [optimization diagnostic report](docs/optimization_diagnostic/README.md) extends training to eight epochs across two data seeds and identifies the next optimization checks.

The [fresh-seed forecasting report](docs/fresh_seed_forecasting/README.md) evaluates eight independent data realizations under a [fixed protocol](docs/fresh_seed_forecasting_protocol.md). Reduced-rank regression has lower shifted-test native forecast MSE than the quantum transition encoder in all eight realizations at this training budget. All model and probe selections were locked before test generation.

The [initialization sensitivity grid](docs/initialization_sensitivity/README.md) checks the native eight-epoch result across two initialization seeds and two learning rates, with training and validation data only.

The [paired initialization control](docs/initialization_control/README.md) isolates the decoder starting-state effect using training and validation only. The larger exploratory configuration explicitly opts into `feature_neutral` quantum initialization; the model default and smoke configuration retain `near_zero`.

The larger exploratory configuration is:

```bash
python -m pilot.run --config configs/cpu_pilot.json --output pilot_runs/replicated
```

It uses three independent data seeds and two initialization seeds, with every learning-rate candidate completing 16 epochs. It is an exploratory pilot, not a powered confirmatory study. Exact density-matrix simulation can be slow. Full runs save datasets, checkpoints, probes, per-sequence measurements, source hashes, package versions and selection diagnostics.

To run regression checks:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

## What changed

- Reconstruction models learn same-time targets; transition models learn next-time targets.
- Classical decoders receive only fixed retained coordinates. Quantum decoders receive the retained reduced state with discarded qubits reset to zero. Recurrent memory follows the same restriction.
- Both families are scored against the same bounded observation features using sample-weighted MSE. Quantum mixed states remain mixed; readout uses expectation values.
- Training and validation choose parameters, checkpoints and probe penalties. Held-out test sequences and shift regimes are evaluated after selection.
- Shared ridge probes test representation usefulness independently of native decoder capacity. PCA, reduced-rank forecasting, MLP, GRU, random projections, persistence and untrained models provide controls.
- Discrete symbol entropy uses normalized symbol counts. Temporal readout descriptors are explicitly distinguished from quantum-state complexity and description length.

Read the [study protocol](docs/study_protocol.md) for assumptions, limitations and the remaining publication work. Old checkpoints are incompatible with protocol 2 and are rejected by corrected model loaders.

## Licensing

Original project material is released under [Zero-Clause BSD](LICENSE), an unrestricted permissive license with no attribution condition. Dependencies retain their upstream licenses. [LICENSE_SCOPE.md](LICENSE_SCOPE.md) explicitly covers the historical code blobs named in issue #4. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the audit and figure replacement.
