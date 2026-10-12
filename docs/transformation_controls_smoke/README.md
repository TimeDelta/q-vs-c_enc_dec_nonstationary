# Transformation-control implementation smoke

All seven new controls run alongside the original Givens (`cte`), native quantum (`qte`), no-entanglement quantum and raw-input reduced-rank models. This uses already inspected diagnostic seed 97, length 64 and two training epochs per learning-rate candidate. It validates implementation and does not supply an architecture ranking or new manuscript finding.

| Model case | Trainable architecture parameters | Readout dimension | Native task |
| --- | ---: | ---: | --- |
| cte | 8 | 2 | forecast |
| untrained_cte | 8 | 2 | forecast |
| qte | 8 | 2 | forecast |
| untrained_qte | 8 | 2 | forecast |
| qte_noent | 8 | 2 | forecast |
| untrained_qte_noent | 8 | 2 | forecast |
| reduced_rank | 0 | 2 | forecast |
| encoding_ridge | 0 | 81 | forecast |
| encoding_reduced_rank | 0 | 2 | forecast |
| cte_affine | 14 | 2 | forecast |
| untrained_cte_affine | 14 | 2 | forecast |
| cte_shared | 16 | 2 | forecast |
| untrained_cte_shared | 16 | 2 | forecast |
| qte_shared | 16 | 2 | forecast |
| untrained_qte_shared | 16 | 2 | forecast |
| qte_ry | 8 | 2 | forecast |
| untrained_qte_ry | 8 | 2 | forecast |
| qte_ry_noent | 8 | 2 | forecast |
| untrained_qte_ry_noent | 8 | 2 | forecast |

All 19 model/probe cases are locked before generating any test partitions. All 228 sequence/task rows are finite and complete. The 44 selection-file hashes remain unchanged. Every candidate completes its epoch budget and validation selects its minimum-scoring checkpoint. Maximum restored native validation error difference: 0.

## Transformation checks

Untrained cases perform no gradient updates despite retaining the architecture parameter count. Analytic references have fitted coefficient arrays and zero gradient-trained parameters.

The bounded-RY product basis has 81 columns. On 17 held-out feature vectors, coefficients derived from 81 interpolation points reproduce the native and retained-readout functions:

| Model | Maximum absolute span error |
| --- | ---: |
| qte | 1.22e-15 |
| qte_noent | 9.99e-16 |
| qte_ry | 1.33e-15 |
| qte_ry_noent | 9.99e-16 |
| qte_shared | 8.88e-16 |

These are numerical implementation checks of the algebraic feature-span derivation, not a forecasting experiment or a fitted performance comparison. The gain/offset Givens extension starts with original predictions to numerical precision. Both shared-decoder architectures have 16 trainable parameters and exactly identical initial head coefficients and offsets.

Source commit: `ed922817655c7be589a2a0008a21356d77aa02f7`. Archive SHA-256: `564dabc4961c6c8690f4e729c56c559fdd4025eeb188d65ac38402a1319ee82f`. Total smoke wall time: 112.6 seconds.

[Protocol and interpretation](../transformation_controls_protocol.md), [verification](verification.json), [configuration](config.json), [selections](fit_diagnostics.json), [metric rows](sequence_metrics.csv) and [all data, model/probe checkpoints and lock files](transformation_controls_assets.zip).

Reproduce from a clean committed checkout with pinned CPU dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.fresh_study --config configs/transform_controls_smoke.json --output /tmp/transformation_controls_smoke
PYTHONPATH=. python scripts/report_transform_controls_smoke.py /tmp/transformation_controls_smoke /tmp/transformation_controls_report
```
