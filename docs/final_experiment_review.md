# Final experiment review

Review date: 2026-10-12. Reviewed implementation and protocol source: [`e479891ab6f38981a030d3889f20e508ecb21087`](https://github.com/TimeDelta/q-vs-c_enc_dec_nonstationary/commit/e479891ab6f38981a030d3889f20e508ecb21087). The source was committed directly to master before the production preflight. The [preflight record](transformation_replication_preflight.json) reports a clean checkout, zero generated datasets and zero fitted models. The full scientific experiment has not started and its fresh training, validation and test observations remain untouched.

**Disposition: ready to execute the specified prospective, fixed-budget transformation benchmark.** This readiness concerns implementation and a defined analysis procedure. It does not establish convergence, equal encoder capacity, broad generalization or readiness to submit the manuscript with new findings. The original Givens model and native quantum model remain unchanged by this review.

## Findings and corrections

| Finding | Consequence before review | Correction |
| --- | --- | --- |
| New controls had only an implementation-smoke configuration | There was no fixed scientific sample, primary contrast or optimization budget to review | Added an untouched eight-data-seed, three-initialization replication, explicit primary endpoint and fixed search before fitting |
| The existing forecasting report hardcodes QTE versus reduced-rank and misclassifies new analytic controls as untrained cases | Applying it to the transformation run would fail coverage verification or report the wrong question | Added a dedicated report driven by the declared primary contrast and actual analytic/trainable model families |
| The runner evaluated the fitted in-memory objects after saving without a pretest restoration gate | Serialization differences could invalidate the archived model or probe evidence | Every checkpoint and frozen probe now replays on validation before lock; test evaluation uses restored objects and saved coefficients |
| The selection-lock file's own digest was recorded only after evaluation | The pretest boundary lacked a fixed identity for the lock itself | Save its digest in the manifest before test access and verify it thereafter |
| Interrupted output lacked an explicit completion phase | Incomplete fitting/evaluation output could be mistaken for a finished run | Record fitting, locked, evaluating and complete phases; the final report requires complete status |
| Configuration accepted duplicate models, invalid block counts and nonfinite finite-difference widths | Checkpoint names could collide or execution could fail after fitting began | Fail before dataset generation; also validate seed exclusions, dimensions, initialization, topology and worker settings |
| Native selection records lacked explicit candidate/epoch identities | A later report had to infer which checkpoint was selected | Record candidate index/value and selected epoch; verify complete budgets, minima and first-tie rules |
| Fixed-budget diagnostics were not specified for the new study | Longer training could still be misrepresented as converged architecture performance | Prespecify training-improvement, late-validation-minimum and grid-boundary diagnostics without altering the run or its sample |

These corrections add no new model family, change no original Givens rotation and introduce no third-party code dependency. Existing checked-in results and manuscript findings were not recomputed or replaced.

## Model and mathematical audit

The original Givens map is rotation, fixed-coordinate projection and rotation. Orthogonality gives its two nonzero singular values as one. The gain/offset extension starts with the original predictions, then permits the retained-coordinate scales and output intercept to change. A single ring block remains a restricted family within the orthogonal group; it is not arbitrary reduced-rank regression.

The primary shared pair has four encoder angles plus eight affine coefficients and four offsets in each model. Its initial head coefficients and offsets agree exactly for all three prospective initialization seeds. Both use the same forward-difference encoder gradients, analytic head gradients and Adam update. Independent central-difference checks verify shared-control gradients. No native decoder is accessible through the shared-head prediction path. Matching this interface and procedure does not match the encoder function classes or their optimization landscapes.

The bounded-RY embedding produces products of `1`, `x_i` and `sqrt(1-x_i**2)`, giving 81 basis columns for four inputs. Because later operations and expectation readouts are linear in the input density matrix, nonrecurrent native circuit predictions and retained Z expectations lie in this span. Numerical interpolation on 81 points reproduces these functions on separately generated feature vectors. This establishes a function-span reference, not parameter-matched coefficient capacity or a guaranteed predictive ranking. The unrestricted encoding-ridge reference is marked as 81-coordinate and uncompressed; persistence is marked as four-coordinate and uncompressed.

The RY-only controls preserve angle counts and real amplitudes. They change gate geometry and retain CZ entanglement when present. They cannot be interpreted as a classical replacement. The no-CZ pair provides the separate entanglement intervention. Original default-family quantum checkpoints remain compatible and archived checkpoints replay under the reviewed implementation.

## Quantum compression semantics

The implementation uses `DensityMatrix.reset` after encoding: discard the trash subsystem and replace it with a zero computational-basis state. IBM documents partial reset as evolution by a reset SuperOp. The implementation preserves the complete retained density matrix, including its mixedness, without postselecting a measurement outcome. Bell-state and four-qubit random-state regressions verify the retained marginal, trace, validity and tensor-product replacement. The four-qubit check covers the actual two-retained/two-discarded setup.

**This is not an outgoing-edge-only mask.** The code does replace the discarded subsystem with `|0><0|` after tracing it out. Entanglement and other correlations between retained and discarded subsystems are removed, as required by this discard channel. Entanglement within the retained subsystem can remain. If preserving the full joint state while masking selected decoder connections is the intended intervention, that would be a different specified model and this protocol would need revision before execution. The present readiness judgment applies to the discard-and-replace benchmark stated in the protocol.

The original native quantum decoder can use the entire retained reduced density matrix. Its two-qubit bottleneck therefore has a different state capacity from two real Givens coordinates. The primary shared-head comparison controls that readout disparity by allowing only two retained Z expectations. This is why the native and shared-head comparisons must remain separately labeled.

## Data, causality and selection audit

Mixing matrices and sequence innovations derive from disjoint seed components. Training and validation reproduce exactly when the test partitions are later generated. All models share the same observations and bounded-feature targets. Stored generating states never enter model fitting, checkpoint selection or probe fitting. Forecast targets are the next observation; reconstruction targets are the current observation. Predictions are one-step and teacher-forced. A future-observation perturbation test checks that earlier native predictions and latent readouts remain identical for every model, including GRU.

Every learning-rate candidate starts from the same architecture-specific seed and completes 64 epochs. Native validation MSE chooses its checkpoint and rate. Probes fit training latents and use validation for ridge selection; they do not choose the native model. Untrained controls retain their exact seeded initialization. Analytic fits are not mislabeled as untrained gradient models. No best initialization is selected for reporting.

After all selections are saved, checkpoint replay, source identity and source/protocol/dependency hashes are checked. All selected files are locked before any test generation. Probe arrays are restored from saved coefficients and marked read-only. A restoration discrepancy, source change or changed hash stops the run. Test scores are never used to choose models, rates, epochs, penalties, sample counts or analysis endpoints.

The report independently restores every selected model and recalculates every native/probe test score and constant-reference score. It checks exact dataset regeneration, row identities, full candidate budgets, selected minima, metadata, finite outputs and lock identity. Synthetic tampering tests confirm that changed scores and lock files are rejected. This verifies numerical consistency of the same saved algorithms; it is not an independent scientific replication.

## Frozen run and interpretation

| Quantity | Fixed specification |
| --- | --- |
| Data seeds | 223, 227, 229, 233, 239, 241, 251, 257 |
| Initialization seeds | 401, 409, 419, crossed with all data seeds |
| Sequence length and splits | 128; four train, four validation and four test sequences per condition |
| Primary contrast | `qte_shared - cte_shared` in native shifted forecast MSE |
| Weighting | Sequence mean, five-condition mean, then mean of all three initializations |
| Independent analysis units | Eight generator realizations |
| Gradient search | 64 epochs each at rates 0.001, 0.005, 0.02 and 0.08 |
| Ridge searches | Seven penalties from 1e-6 to 1, fit on training and selected on validation |
| Models and untrained controls | 16 named models plus ten untrained cases per data/initialization group |
| Planned totals | 624 model/probe cases, 29,952 sequence/task rows, 384 constant rows, 61,440 gradient updates |
| Execution | Four isolated CPU fitting processes, one Torch thread each, exact density matrices, no shots |

The primary score gives equal weight to all five shifted conditions and all initialization seeds. Eight realization-level differences and each initialization-specific difference are retained. More timesteps, related sequences or initialization seeds do not increase the number of independent data realizations. There is no specified significance threshold or power claim.

Optimization diagnostics flag a greater-than-1% reduction between the last two 16-epoch median training-loss windows, a validation minimum in the final window and selected grid boundaries. These flags do not change the experiment. Failure to flag continuing improvement does not prove convergence. The primary result remains a comparison of the fixed procedure.

The generator is still one known, bounded, piecewise Gaussian AR family with fixed regime timing. Conditions use independent innovations, so condition differences include sampling variation. Changes in persistence preserve within-regime stationary variance but transient regimes need not be stationary. The legacy variance-ratio field changes standard deviation. This experiment supplies no recursive forecasting, hardware quantum, computational speedup, clinical or neuroimaging evidence. Another substantially different generator or real dataset is required for broader generalization claims.

Coordinatewise temporal summaries remain dependent on representation and are exploratory here. Rotation compensation preserves probe predictions while descriptors can change. This replication addresses transformation and readout disparity; it does not provide a confirmatory descriptor-error association or an MDL estimator. Broad related-work and manuscript author review remain separate publication requirements.

## Validation and execution boundary

All 48 local regression tests passed in 81.358 seconds. Ten review regressions add data-free preflight checks, causality, four-qubit marginal preservation, correct grouped aggregation, optimization diagnostics, early checkpoint/source rejection, complete control-report replay, tampering rejection and shared-control parallel/sequential agreement. Earlier 38 tests continue to pass, including gradient, feature-span and checkpoint compatibility checks. Compilation and whitespace checks passed. Tests used diagnostic fixtures; the prospective study did not run.

The [GitHub correctness run for the committed source](https://github.com/TimeDelta/q-vs-c_enc_dec_nonstationary/actions/runs/38283462968) also passed.

The production preflight ran from the clean published source and verified all 12 native quantum calibrations, every initialization's identical shared heads and the planned model/output counts. Its JSON record is checked in separately from scientific results. Core dependencies are pinned; relevant transitive dependency versions and BLAS/OpenMP settings are recorded in run manifests.

Use the exact [prospective configuration](../configs/transformation_replication.json) and [run/report commands](transformation_replication_protocol.md). The mandatory preflight also runs inside the scientific runner. A new output directory is required. No full experiment was launched as part of this review.

## Supporting primary sources

- [IBM Qiskit DensityMatrix.reset](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.DensityMatrix): "Reset state or subsystems to the 0-state." Its partial-reset description specifies reset-channel evolution.
- [Schuld (2021)](https://arxiv.org/abs/2101.11020), *Supervised quantum machine learning models are kernel methods*. The repository-specific finite feature span is derived from the implemented embedding and checked numerically.
- [Bowles, Ahmed and Schuld (2024)](https://arxiv.org/abs/2403.07059), *Better than classical? The subtle art of benchmarking quantum machine learning models*. Supports keeping strong classical references and distinguishing selection, feature maps and entanglement interventions.
