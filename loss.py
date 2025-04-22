from qiskit.quantum_info import Statevector, state_fidelity, partial_trace, DensityMatrix

from utility import *
from models import *

def trash_qubit_penalty(state, bottleneck_size):
    """
    Compute a penalty based on the marginal probability of each qubit being in |0>.

    For each qubit in the state, compute P(|0>), then sort these probabilities in ascending order.
    Treat the lowest (num_qubits - bottleneck_size) probabilities as the trash qubits and define
    the penalty as the sum over these of (1 - P(|0>)).

    Parameters:
        state (Statevector): state from which to compute marginals
        bottleneck_size (int): # qubits reserved for latent space

    Returns:
        penalty (float): cumulative penalty for the trash qubits
    """
    marginals = []

    # compute marginal probability for each qubit being in |0>
    for q in range(state.num_qubits):
        trace_indices = list(range(state.num_qubits))
        trace_indices.remove(q)
        density_matrix = partial_trace(state, trace_indices)
        # probability that qubit q = |0> is (0, 0) entry of reduced density matrix
        pq = np.real(density_matrix.data[0, 0])
        marginals.append(pq)

    sorted_marginals = sorted(marginals)
    num_trash = state.num_qubits - bottleneck_size

    # marginals are probabilities that qubit is in |0>
    return sum(1 - p for p in sorted_marginals[:num_trash])

def classical_trash_penalty(state, bottleneck_size):
    num_trash = state.shape[0] - bottleneck_size
    # use the lowest magnitude features as trash
    return sum(np.sort(np.abs(state))[:num_trash])

def autoregressive_cost_function(trash_penalty_fn) -> float:
    return lambda data, model, trash_penalty_weight: main_cost_function(data, model, trash_penalty_fn, trash_penalty_weight, autoregressive=True)

def autoencoder_cost_function(trash_penalty_fn) -> float:
    return lambda data, model, trash_penalty_weight: main_cost_function(data, model, trash_penalty_fn, trash_penalty_weight, autoregressive=False)

def main_cost_function(data, model, trash_penalty_fn, trash_penalty_weight=1, autoregressive=False) -> float:
    total_prediction_cost = 0.0
    total_trash_cost = 0.0
    num_predictions = 0
    num_close_bottlenecks = 0
    num_close_predictions = 0
    num_close_costs = 0
    threshold = .001
    for (_, series) in data:
        model.reset_hidden_state()

        series_prediction_cost = 0.0
        series_trash_cost = 0.0

        prev_bottleneck = None
        prev_prediction = None
        prev_prediction_cost = None
        num_examples = len(series)
        if autoregressive:
            num_examples -= 1
        for t in range(num_examples):
            input_state = model.prepare_state(series[t])
            if autoregressive:
                ideal_state = model.prepare_state(series[t+1])
            else:
                ideal_state = input_state

            # pass the (possibly perturbed) input state through the encoder and decoder
            bottleneck_state, predicted_state = model.forward(input_state)
            if prev_bottleneck is not None:
                dist = np.linalg.norm(prev_bottleneck - bottleneck_state)
                if dist < threshold:
                    num_close_bottlenecks += 1
            prev_bottleneck = bottleneck_state

            if prev_prediction is not None:
                dist = np.linalg.norm(prev_prediction - predicted_state)
                if dist < threshold:
                    num_close_predictions += 1
            prev_prediction = predicted_state

            series_trash_cost += trash_penalty_fn(bottleneck_state, model.bottleneck_size)

            prediction_cost = np.linalg.norm(ideal_state - predicted_state)
            if prev_prediction_cost is not None:
                dist = (prev_prediction_cost - prediction_cost)**2
                if dist < threshold:
                    num_close_costs += 1
            prev_prediction_cost = prediction_cost
            series_prediction_cost += prediction_cost

        # normalize series costs by number of predictions
        # same number of series per dataset
        num_predictions += len(series)
        if autoregressive:
            num_predictions -= 1
        total_trash_cost += series_trash_cost / num_predictions
        total_prediction_cost += series_prediction_cost / num_predictions

    total_trash_cost *= trash_penalty_weight
    print('      percentage of close consecutive bottlenecks:', num_close_bottlenecks/num_predictions)
    print('      percentage of close consecutive predictions:', num_close_predictions/num_predictions)
    print('      percentage of close consecutive prediction costs:', num_close_costs/num_predictions)
    print('      prediction cost:', total_prediction_cost)
    print('      trash cost:', total_trash_cost)
    return [total_prediction_cost, total_trash_cost]
