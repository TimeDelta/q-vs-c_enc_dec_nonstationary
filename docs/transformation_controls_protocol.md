# Transformation controls alongside the original Givens model

The original `cte` Givens encoder/decoder and `qte` quantum encoder/decoder remain separate, unchanged default models. The new controls address specific differences in encoding, native transformation restrictions and decoder access. Equal angle counts do not establish equal function classes. Results from the existing forecasting study and manuscript are not retroactively replaced by new controls.

## Models and controlled differences

| Name | Construction | Purpose | Gradient parameters for four inputs, two retained features and one block |
| --- | --- | --- | ---: |
| `cte` | Original Givens rotations, fixed-coordinate projection and Givens decoding | Original classical comparison | 8 |
| `qte` | Original tied Rx/Ry/Rz rotations with CZ, discard/reset and quantum decoding | Original quantum comparison | 8 |
| `cte_affine` | Original Givens rotations plus two decoder-input gains and four output offsets | Remove unit-singular-value and zero-offset restrictions | 14 |
| `cte_shared` | Original Givens encoder, two retained coordinates and affine head | Shared decoder-access control | 16 |
| `qte_shared` | Original quantum encoder, two retained Z expectations and the same affine-head architecture | Shared decoder-access control | 16 |
| `qte_ry` | Tied pre/post Ry rotations with the original CZ topology | Real-amplitude circuit ablation | 8 |
| `qte_ry_noent` | Same Ry-only circuit without CZ gates | Real-amplitude/no-entanglement factorial control | 8 |
| `encoding_ridge` | Explicit bounded-RY product features and ridge forecast coefficients | Encoding-feature reference with no additional bottleneck | 0 |
| `encoding_reduced_rank` | Same product features and rank-two forecast regression | Encoding-feature reference with a two-coordinate linear bottleneck | 0 |

Zero gradient parameters means analytic fitting, not zero capacity. The encoding references have more fitted coefficients than the circuit. The RY-only ablation remains a quantum circuit and can remain entangled when CZ gates are present. It changes gate structure and optimization geometry despite matching angle counts; it is not an assertion of equal overall expressive capacity.

### Givens gains and offsets

For the original nonrecurrent map, `R_decoder P R_encoder` has two nonzero singular values equal to one. The added diagonal gains act on retained coordinates before padding with zeros and applying the existing decoder rotations. Four offsets are added afterward. At unit gains and zero offsets the model exactly reproduces the original Givens outputs for the same seeded angles. The encoder readout used by common probes remains the original two coordinates, before decoder gains. Gains and offsets are trained by the existing classical autograd/Adam procedure.

This extends the original model without replacing its rotations. One ring block is still more restricted than arbitrary orthogonal encoder/decoder matrices, so this does not turn it into general reduced-rank regression.

### Shared-decoder training

The shared pair is trained end to end from its two real readouts, rather than selecting encoders through different native decoders and fitting probes afterward. Both use an affine head with a 2-by-4 coefficient matrix and four offsets. Both start with exactly the same seeded head coefficients and zero offsets. Each encoder retains its original near-zero seeded angles. Native quantum decoder calibration is omitted because that decoder is inaccessible and replaced by the affine head.

Only encoder angles and affine-head parameters are optimized: four plus twelve in each model. Quantum features are retained single-qubit Z expectations; classical features are retained Givens coordinates. The head receives neither the complete density matrix nor discarded-qubit information. Both families use forward-difference encoder gradients, analytic affine-head gradients and the same Adam update rule. Every learning-rate candidate completes the same epoch budget; native validation forecasting MSE selects its checkpoint and rate. Equality of this interface and procedure does not remove encoder inductive-bias, feature-range or optimization-landscape differences.

The shared interface converts the quantum representation to two classical numbers. It tests that restricted measurement channel, whereas the original native quantum decoder retains access to the complete reduced state. These are different experiments and both remain available.

### Explicit encoding-feature references

Bounded RY embedding gives each input qubit density matrix `(I + sqrt(1-x_i**2) X + x_i Z)/2`. The product input state therefore expands in products of `1`, `x_i` and `sqrt(1-x_i**2)`. For four inputs there are 81 feature columns, including the constant column.

All subsequent data-independent unitary operations, partial discard/reset channels and expectation readouts are linear in the input density matrix. Native nonrecurrent predictions and retained Z readouts consequently lie in this feature span. Quantum circuit parameters constrain their coefficients. A two-Z-readout model followed by an affine head has a feature-coefficient matrix of rank at most two, apart from its output intercept. The rank-two reference uses the same feature basis. Neither coefficient freedom nor regularization is parameter-matched to the circuit, and better generalization is not guaranteed.

The derivation applies to a single data embedding per observation without recurrence. Reuploading inputs or feeding prior states into later predictions requires a different feature-span analysis. Explicit expansion is restricted to at most six input features to bound its exponential storage and computation. Input and target means are estimated from training only and validation selects ridge regularization using the existing candidate mechanism. `encoding_ridge` retains all 81 feature coordinates and is explicitly uncompressed; its reconstruction probe and descriptor scores must not be presented as those of a two-coordinate bottleneck.

The feature-map perspective is consistent with Schuld, *Supervised quantum machine learning models are kernel methods* (2021), <https://arxiv.org/abs/2101.11020>. The exact 81-column construction above is derived from this repository's embedding, with numerical span checks for both retained and native readouts. Model selection and hyperparameter fairness also require attention independently of feature matching; see Bowles, Ahmed and Schuld (2024), <https://arxiv.org/abs/2403.07059>.

## Verification and implementation smoke

Regression checks require exact reproduction of original Givens outputs at unit gains/zero offsets, nonzero gradients of the added parameters, identical initial shared heads, agreement with independent numerical gradients, no native-decoder access in shared predictions, complete feature-span reconstruction on held-out inputs, real amplitudes in the Ry ablation and checkpoint roundtrips for every new family. Existing archived default-family quantum checkpoints must still restore. Default checkpoint metadata remains compatible; RY checkpoints carry their nondefault family and reject mismatched loaders.

The committed `configs/transform_controls_smoke.json` runs all seven new controls alongside original `cte`, `qte`, `qte_noent` and raw-input reduced-rank regression. It uses already inspected seed 97, initialization 303, length 64, two training sequences, one validation sequence, one test sequence per condition and two epochs per learning-rate candidate. Untrained controls are included. This is an implementation smoke check and cannot rank architectures or confirm a scientific hypothesis.

The two-stage runner fits and hash-locks all models and probes before generating its test partitions. After the source is committed, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pilot.fresh_study --config configs/transform_controls_smoke.json --output /tmp/transformation_controls_smoke
```

The smoke uses two learning rates, 0.02 and 0.08 and native ridge penalties 0.001 and 0.01. It establishes executable selection, complete finite outputs and lock invariance. The broad selection range and convergence budgets needed for scientific comparisons are not established by two epochs.

## Subsequent scientific replication

A new protocol should fix independent data and initialization seeds, convergence diagnostics, candidate budgets and endpoints before fitting. The central matched-interface comparison should retain `cte_shared` versus `qte_shared`, with the original Givens/native quantum contrast, gain/offset intervention, encoding references and phase/no-entanglement controls reported separately. Neither old inspected test outcomes nor this smoke seed should become a new confirmatory test sample. Such an experiment has not been run in this implementation phase.
