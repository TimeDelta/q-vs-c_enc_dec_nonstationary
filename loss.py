"""Protocol 2 losses: feature-space MSE and a fixed-subsystem trash penalty."""

import numpy as np
from qiskit.quantum_info import partial_trace


def trash_qubit_penalty(state, bottleneck_size):
    """Mean excitation probability of the fixed discarded qubits."""
    trash_probabilities = []
    for qubit in range(bottleneck_size, state.num_qubits):
        trace_indices = [index for index in range(state.num_qubits) if index != qubit]
        reduced_state = partial_trace(state, trace_indices)
        trash_probabilities.append(1 - float(np.real(reduced_state.data[0, 0])))
    return float(np.mean(trash_probabilities)) if trash_probabilities else 0.0


def classical_trash_penalty(state, bottleneck_size):
    discarded_features = np.asarray(state)[bottleneck_size:]
    return float(np.mean(np.square(discarded_features))) if discarded_features.size else 0.0


def autoregressive_cost_function(trash_penalty_fn):
    return lambda data, model, trash_penalty_weight: main_cost_function(
        data, model, trash_penalty_fn, trash_penalty_weight, autoregressive=True
    )


def autoencoder_cost_function(trash_penalty_fn):
    return lambda data, model, trash_penalty_weight: main_cost_function(
        data, model, trash_penalty_fn, trash_penalty_weight, autoregressive=False
    )


def cost_function_for_model_type(model_type, trash_penalty_fn):
    if model_type not in ('qae', 'qrae', 'qte', 'qrte', 'cae', 'crae', 'cte', 'crte'):
        raise ValueError('Unknown model type: ' + model_type)
    factory = autoregressive_cost_function if model_type.endswith('te') else autoencoder_cost_function
    return factory(trash_penalty_fn)


def main_cost_function(data, model, trash_penalty_fn, trash_penalty_weight=1, autoregressive=False):
    """Average squared error over all evaluated timesteps and feature coordinates.

    Targets always remain in the original feature space. Quantum outputs are
    read from the complete predicted density matrix, including mixed states.
    Sequence boundaries reset recurrence. There is no free-running forecast here.
    """
    if trash_penalty_weight < 0 or not np.isfinite(trash_penalty_weight):
        raise ValueError('trash_penalty_weight must be finite and nonnegative')
    prediction_cost_sum = 0.0
    trash_cost_sum = 0.0
    evaluated_steps = 0
    for _, series in data:
        series = np.asarray(series)
        if series.ndim != 2 or not np.all(np.isfinite(series)):
            raise ValueError('Expected a finite time-by-feature array')
        model.reset_hidden_state()
        sequence_steps = len(series) - int(autoregressive)
        if sequence_steps < 1:
            raise ValueError('Every sequence must contain at least one evaluated target')
        for timestep in range(sequence_steps):
            bottleneck_state, predicted_state = model.forward(model.prepare_state(series[timestep]))
            prediction = model.collapse_state(predicted_state) if hasattr(model, 'collapse_state') else predicted_state
            target = series[timestep + int(autoregressive)]
            prediction = np.asarray(prediction)
            if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
                raise ValueError('Predictions must be finite and match the target feature shape')
            prediction_cost_sum += float(np.mean(np.square(target - prediction)))
            trash_cost_sum += trash_penalty_fn(bottleneck_state, model.bottleneck_size)
            evaluated_steps += 1
    if not evaluated_steps:
        raise ValueError('Cannot evaluate an empty partition')
    return [prediction_cost_sum / evaluated_steps,
            trash_penalty_weight * trash_cost_sum / evaluated_steps]
