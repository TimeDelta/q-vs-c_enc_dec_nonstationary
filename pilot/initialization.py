"""Data-free calibration of existing quantum decoder parameters.

No gates or parameters are added. Only the first decoder block's discarded
qubit rotation parameters are recentered; the original seed jitter is retained.
"""
import time
import numpy as np
from scipy.optimize import least_squares


def feature_neutral_parameters(model, parameter_handles, parameter_values):
    if model.feature_encoding != 'bounded_ry' or not model.enforce_bottleneck:
        raise ValueError('Feature-neutral initialization requires bounded encoding and enforced compression')
    started_at = time.perf_counter()
    original_values = np.asarray(parameter_values).copy()
    names = {f'Decoder Pre-Layer 0 Rθ {qubit}' for qubit in range(model.bottleneck_size, model.num_qubits)}
    indices = [index for index, parameter in enumerate(parameter_handles) if parameter.name in names]
    if len(indices) != model.num_qubits - model.bottleneck_size or not indices:
        raise ValueError('Expected one existing first-block decoder parameter per discarded qubit')
    reference_state = model.prepare_state(np.zeros(model.num_qubits))
    evaluation_count = 0

    def residual(candidate_values):
        nonlocal evaluation_count
        evaluation_count += 1
        complete_values = original_values.copy()
        complete_values[indices] = candidate_values
        model.set_params(dict(zip(parameter_handles, complete_values)))
        model.reset_hidden_state()
        _, prediction = model.forward(reference_state)
        return model.collapse_state(prediction)[model.bottleneck_size:]

    initial_readout = residual(original_values[indices])
    calibration = least_squares(residual, np.full(len(indices), np.pi/4), bounds=(-np.pi, np.pi),
                                ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=128)
    center_error = float(np.max(np.abs(calibration.fun)))
    if not calibration.success or center_error > 1e-8:
        model.set_params(dict(zip(parameter_handles, original_values)))
        model.reset_hidden_state()
        raise ValueError('Feature-neutral calibration did not reach the reference readout tolerance')
    calibrated_values = original_values.copy()
    # Keep the same original seed jitter instead of initializing at a stationary
    # symmetry point. No training, validation or test observations enter calibration.
    calibrated_values[indices] = calibration.x + original_values[indices]
    final_readout = residual(calibrated_values[indices])
    model.reset_hidden_state()
    metadata = dict(reference='zero observation vector; no dataset access',
                    changed_parameter_indices=indices,
                    changed_parameter_names=[parameter_handles[index].name for index in indices],
                    calibration_centers=calibration.x.tolist(),
                    initial_reference_readout=initial_readout.tolist(),
                    center_max_abs_readout=center_error,
                    jittered_reference_readout=final_readout.tolist(),
                    calibration_evaluations=evaluation_count,
                    calibration_seconds=time.perf_counter()-started_at)
    return calibrated_values, metadata
