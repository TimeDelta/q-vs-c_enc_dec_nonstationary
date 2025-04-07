import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace, DensityMatrix, Operator, random_statevector
import re

from utility import soft_reset_trash_qubits, force_trash_qubits, dm_to_statevector, without_t_gate

ENTANGLEMENT_OPTIONS = ['full', 'linear', 'circular']
ENTANGLEMENT_GATES = ['cx', 'cz', 'rzx']

def create_embedding_circuit(num_qubits, embedding_gate, include_time_step=False):
    """
    Apply rotation gate to each qubit for embedding of classical data.
    Returns qc, input_params
    """
    input_params = []
    num_qubits = num_qubits+1 if include_time_step else num_qubits
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if include_time_step and i == num_qubits-1:
            p = Parameter('t')
        else:
            p = Parameter('Embedding Rθ ' + str(i))
        input_params.append(p)
        if embedding_gate.lower() == 'rx':
            qc.rx(p, i)
        elif embedding_gate.lower() == 'ry':
            qc.ry(p, i)
        elif embedding_gate.lower() == 'rz':
            qc.rz(p, i)
        else:
            raise Exception("Invalid embedding gate: " + embedding_gate)
    return qc, input_params

def add_entanglement_topology(qc, num_qubits, entanglement_topology, entanglement_gate):
    if entanglement_topology == 'full':
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                if entanglement_gate.lower() == 'cx':
                    qc.cx(i, j)
                elif entanglement_gate.lower() == 'cz':
                    qc.cz(i, j)
                elif entanglement_gate.lower() == 'rzx':
                    qc.rzx(np.pi/4, i, j)
                else:
                    raise Exception("Unknown entanglement gate: " + entanglement_gate)
    elif entanglement_topology == 'linear':
        for i in range(num_qubits - 1):
            if entanglement_gate.lower() == 'cx':
                qc.cx(i, i+1)
            elif entanglement_gate.lower() == 'cz':
                qc.cz(i, i+1)
            elif entanglement_gate.lower() == 'rzx':
                qc.rzx(np.pi/4, i, i+1)
            else:
                raise Exception("Unknown entanglement gate: " + entanglement_gate)
    elif entanglement_topology == 'circular':
        for i in range(num_qubits - 1):
            if entanglement_gate.lower() == 'cx':
                qc.cx(i, i+1)
            elif entanglement_gate.lower() == 'cz':
                qc.cz(i, i+1)
            elif entanglement_gate.lower() == 'rzx':
                qc.rzx(np.pi/4, i, i+1)
            else:
                raise Exception("Unknown entanglement gate: " + entanglement_gate)
        if entanglement_gate.lower() == 'cx':
            qc.cx(num_qubits-1, 0)
        elif entanglement_gate.lower() == 'cz':
            qc.cz(num_qubits-1, 0)
        elif entanglement_gate.lower() == 'rzx':
            qc.rzx(np.pi/4, num_qubits-1, 0)
        else:
            raise Exception("Unknown entanglement gate: " + entanglement_gate)

def create_qed_circuit(bottleneck_size, num_qubits, num_blocks, entanglement_topology, entanglement_gate, include_time_step=False):
    """
    Build a parameterized encoder using multiple layers.

    For each block, we perform:
      - A layer of single-qubit rotations (we use Ry for simplicity).
      - An entangling layer whose connectivity is determined by entanglement_topology.

    The total number of parameters is:
      num_blocks * (2 * num_qubits)
    (Each layer has 2 * num_qubits parameters: one set before the entangling layer and one set after.)

    The decoder is taken as the inverse of the entire encoder.
    Returns qte_circuit, encoder, decoder_circuit, all_parameters
    """
    if include_time_step:
        num_qubits += 1
    encoder = QuantumCircuit(num_qubits)
    params = []
    for layer in range(num_blocks):
        # for i in range(num_qubits):
        #     encoder.rx(Parameter('Encoder Layer ' + str(layer) + ' Rx θ ' + str(i)), i)
        add_entanglement_topology(encoder, num_qubits, entanglement_topology, entanglement_gate)
        for i in range(num_qubits):
            encoder.ry(Parameter('Encoder Layer ' + str(layer) + ' Ry θ ' + str(i)), i)
    params.extend(encoder.parameters)

    # For a proper autoencoder, you want to compress information into a bottleneck but
    # choosing which qubits to force to |0> reduces the flexibility the AE has in
    # compressing and reconstructing the data. Instead, in this design, the circuit
    # structure (and subsequent training loss) is expected to force the encoder to
    # focus its information into 'bottleneck_size' qubits.
    decoder = QuantumCircuit(num_qubits)
    # leave the time step qubit in the circuit but don't add any gates to it
    if include_time_step:
        num_qubits -= 1
    for layer in range(num_blocks):
        # for i in range(num_qubits):
        #     decoder.rx(Parameter('Decoder Layer ' + str(layer) + ' Rx θ ' + str(i)), i)
        add_entanglement_topology(decoder, num_qubits, entanglement_topology, entanglement_gate)
        for i in range(0, num_qubits):
            decoder.ry(Parameter('Decoder Layer ' + str(layer) + ' Ry θ ' + str(i)), i)
    params.extend(decoder.parameters)
    full_circuit = encoder.compose(decoder)
    return full_circuit, encoder, decoder, params

def create_full_circuit(num_qubits, config, include_time_step=False):
    """
    config is dict w/ additional hyperparameters:
        entanglement_topology:   options are ['full', 'linear', 'circular']
        entanglement_gate:       options are ['CX', 'CZ', 'RZX']
        penalty_weight:          weight for the bottleneck penalty term.
    Returns full_circuit, encoder_circuit, decoder_circuit, input_params, trainable_params
    """
    embedding_qc, input_params = create_embedding_circuit(num_qubits, config.get('embedding_gate', 'rz'), include_time_step)

    num_blocks = config.get('num_blocks', 1)
    ent_topology = config.get('entanglement_topology', 'full')
    ent_gate_type = config.get('entanglement_gate', 'cx')
    bottleneck_size = config.get('bottleneck_size', int(num_qubits/2))
    qte_circuit, encoder, decoder, trainable_params = create_qed_circuit(
        bottleneck_size, num_qubits, num_blocks, ent_topology, ent_gate_type, include_time_step
    )
    return embedding_qc.compose(qte_circuit), embedding_qc, encoder, decoder, input_params, trainable_params

def avg_per_qubit_fidelity(ideal_state, reconstructed_state):
    """
    1 - (average fidelity over individual qubits)
    """
    if not isinstance(ideal_state, DensityMatrix):
        ideal_state = DensityMatrix(ideal_state)
    if not isinstance(reconstructed_state, DensityMatrix):
        reconstructed_state = DensityMatrix(reconstructed_state)

    num_qubits = int(np.log2(ideal_state.dim))

    fidelities = []
    for i in range(num_qubits):
        trace_out = [j for j in range(num_qubits) if j != i]
        reduced_ideal = partial_trace(ideal_state, trace_out)
        reduced_recon = partial_trace(reconstructed_state, trace_out)
        fidel = state_fidelity(reduced_ideal, reduced_recon)
        fidelities.append(fidel)

    return 1 - np.mean(fidelities)

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
    penalty = sum(1 - p for p in sorted_marginals[:num_trash])
    return penalty

def qae_cost_function(data, embedder, encoder, decoder, input_params, bottleneck_size, trash_qubit_penalty_weight=1, no_noise_prob=1.0, include_time_step=False) -> float:
    """
    With probability (1-no_noise_prob), the input state is perturbed by adding (or subtracting) the error vector
    from the previous time step. The perturbed state is both the input to the encoder and the reconstruction target.

    (1 - average_per_qubit_fidelity(current_state, reconstructed_state)) + trash_qubit_penalty_weight * trash_qubit_penalty()
    """
    total_reconstruction_cost = 0.0
    total_trash_cost = 0.0
    total_num_states = 0
    num_close_bottlenecks = 0
    num_close_predictions = 0
    num_close_costs = 0
    threshold = .001
    for (_, series) in data:
        series_reconstruction_cost = 0.0
        series_trash_cost = 0.0
        previous_error_vector = None
        prev_bottleneck = None
        prev_reconstruction = None
        prev_reconstruction_cost = None
        for t, state in enumerate(series):
            if include_time_step:
                params = {p: state[i] for i, p in enumerate(input_params[:-1])} # final param is time step
                t_param = input_params[-1]
                params[t_param] = t/len(series)
            else:
                params = {p: state[i] for i, p in enumerate(input_params)}
            input_qc = embedder.assign_parameters(params)
            input_state = Statevector.from_instruction(input_qc)
            input_array = input_state.data
            if include_time_step:
                params.pop(t_param)
                ideal_state = Statevector.from_instruction(without_t_gate(embedder).assign_parameters(params))
            else:
                ideal_state = input_state

            if previous_error_vector is not None and np.random.rand() < (1-no_noise_prob):
                sign = 1 if np.random.rand() < 0.5 else -1
                perturbed_array = input_array + sign * previous_error_vector
                # renormalize the perturbed vector to ensure valid quantum state
                perturbed_array = perturbed_array / np.linalg.norm(perturbed_array)
                input_state = Statevector(perturbed_array)
            else:
                input_state = input_state

            # pass the (possibly perturbed) input state through the encoder and decoder
            bottleneck_state = input_state.evolve(encoder)
            if prev_bottleneck is not None:
                dist = np.linalg.norm(prev_bottleneck - bottleneck_state)
                if dist < threshold:
                    num_close_bottlenecks += 1
            prev_bottleneck = bottleneck_state

            trash_cost = trash_qubit_penalty(bottleneck_state, bottleneck_size)
            series_trash_cost += trash_cost

            reconstructed_state = bottleneck_state.evolve(decoder)
            # remove time step qubit from reconstruction
            if include_time_step:
                reconstructed_state = partial_trace(reconstructed_state, [len(embedder.qubits)-1])
            if reconstructed_state.data.ndim > 1:
                reconstructed_state = dm_to_statevector(reconstructed_state)
            if prev_reconstruction is not None:
                dist = np.linalg.norm(prev_reconstruction - reconstructed_state)
                if dist < threshold:
                    num_close_predictions += 1
            prev_reconstruction = reconstructed_state

            reconstruction_cost = np.linalg.norm(ideal_state.data - reconstructed_state.data)
            if prev_reconstruction_cost is not None:
                dist = (prev_reconstruction_cost - reconstruction_cost)**2
                if dist < threshold:
                    num_close_costs += 1
            prev_reconstruction_cost = reconstruction_cost
            series_reconstruction_cost += reconstruction_cost

            previous_error_vector = ideal_state.data - reconstructed_state.data

        # compute avg costs per state (always same num states per series but QTE uses transitions)
        total_trash_cost += series_trash_cost / len(series)
        total_num_states += len(series)
        total_reconstruction_cost += series_reconstruction_cost / len(series)

    total_trash_cost *= trash_qubit_penalty_weight
    print('      percentage of close consecutive bottlenecks:', num_close_bottlenecks/total_num_states)
    print('      percentage of close consecutive predictions:', num_close_predictions/total_num_states)
    print('      percentage of close consecutive reconstruction costs:', num_close_costs/total_num_states)
    print('      reconstruction cost:', total_reconstruction_cost)
    print('      trash qubit cost:', total_trash_cost)
    return [total_reconstruction_cost, total_trash_cost]

def qte_cost_function(data, embedder, encoder, decoder, input_params, bottleneck_size, trash_qubit_penalty_weight=1, teacher_forcing_prob=1.0, dummy_var=False) -> float:
    """
    Scheduled sampling is incorporated via teacher_forcing_prob:
      - With probability teacher_forcing_prob, the next input state is taken from the ground truth.
      - Otherwise, the model's predicted state is used as the next input.

    (1 - fidelity(next_state, predicted_state)) + trash_qubit_penalty_weight * trash_qubit_penalty()
    """
    total_reconstruction_cost = 0.0
    total_trash_cost = 0.0
    total_num_transitions = 0
    num_close_bottlenecks = 0
    num_close_predictions = 0
    num_close_costs = 0
    threshold = .001
    for (_, series) in data:
        series_length = len(series)
        if series_length < 2:
            continue # can't model transition
        series_reconstruction_cost = 0.0
        series_trash_cost = 0.0

        # always use ground truth for first time step
        initial_params = {p: series[0][i] for i, p in enumerate(input_params)}
        qc_initial = embedder.assign_parameters(initial_params)
        current_state = Statevector.from_instruction(qc_initial)

        prev_bottleneck = None
        prev_prediction = None
        prev_reconstruction_cost = None
        for i in range(series_length - 1):
            next_state = series[i+1]
            next_input_params = {p: next_state[j] for j, p in enumerate(input_params)}
            next_state = Statevector.from_instruction(embedder.assign_parameters(next_input_params))

            bottleneck_state = current_state.evolve(encoder)
            if prev_bottleneck is not None:
                dist = np.linalg.norm(prev_bottleneck - bottleneck_state)
                if dist < threshold:
                    num_close_bottlenecks += 1
            prev_bottleneck = bottleneck_state

            trash_cost = trash_qubit_penalty(bottleneck_state, bottleneck_size)
            series_trash_cost += trash_cost

            predicted_state = bottleneck_state.evolve(decoder)
            if predicted_state.data.ndim > 1:
                predicted_state = dm_to_statevector(predicted_state)
            if prev_prediction is not None:
                dist = np.linalg.norm(prev_prediction - predicted_state)
                if dist < threshold:
                    num_close_predictions += 1
            prev_prediction = predicted_state

            reconstruction_cost = np.linalg.norm(next_state.data - predicted_state.data)
            if prev_reconstruction_cost is not None:
                dist = (prev_reconstruction_cost - reconstruction_cost)**2
                if dist < threshold:
                    num_close_costs += 1
            prev_reconstruction_cost = reconstruction_cost
            series_reconstruction_cost += reconstruction_cost

            if np.random.rand() < teacher_forcing_prob:
                current_state = next_state
            else:
                current_state = predicted_state

        # compute avg costs per transition (always same num transitions per series but QAE uses states)
        total_trash_cost += series_trash_cost / (len(series)-1)
        total_num_transitions += len(series)-1
        total_reconstruction_cost += series_reconstruction_cost / (series_length-1)
    total_trash_cost *= trash_qubit_penalty_weight
    print('      percentage of close consecutive bottlenecks:', num_close_bottlenecks/total_num_transitions)
    print('      percentage of close consecutive predictions:', num_close_predictions/total_num_transitions)
    print('      percentage of close consecutive reconstruction costs:', num_close_costs/total_num_transitions)
    print('      reconstruction cost:', total_reconstruction_cost)
    print('      trash qubit cost:', total_trash_cost)
    return [total_reconstruction_cost, total_trash_cost]

def adam_update(params, gradients, moment1, moment2, t, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    moment1 = beta1 * moment1 + (1 - beta1) * gradients
    moment2 = beta2 * moment2 + (1 - beta2) * (gradients ** 2)
    bias_corrected_moment1 = moment1 / (1 - beta1 ** t)
    bias_corrected_moment2 = moment2 / (1 - beta2 ** t)
    new_params = params - lr * bias_corrected_moment1 / (np.sqrt(bias_corrected_moment2) + epsilon)
    return new_params, moment1, moment2

def train_adam(training_data, validation_data, cost_function, config, num_epochs=100, include_time_step=False):
    """
    Train the QTE by minimizing the cost function using ADAM. Note that the QTE will only enforce
    the bottleneck via the cost function. This is done in order to balance efficiency w/ added
    flexibility for which qubits get thrown away.

    Parameters:
      - cost_function: the function to use for calculating the cost

      - config: dict containing additional hyperparameters:
            num_blocks:              # [entanglement layer, rotation layer] repetitions per 1/2 of QTE
            entanglement_topology:   for all entanglement layers
            entanglement_gate:       options are ['CX', 'CZ', 'RZX']
            learning_rate:
            bottleneck_size:         number of qubits for the latent space
            penalty_weight:          weight for the bottleneck penalty term
      - num_epochs: number of training iterations

    Returns trained_circuit, cost_history, validation_costs, embedder, encoder, input_params
    """
    num_qubits = len(training_data[0][1][0])
    cost_history = []
    gradient_width = 1e-4

    bottleneck_size = int(config['bottleneck_size'])
    num_blocks = int(config['num_blocks'])
    learning_rate = float(config['learning_rate'])
    max_penalty_weight = float(config.get('max_penalty_weight', 1.0))

    trained_circuit, embedder, encoder, decoder, input_params, trainable_params = create_full_circuit(num_qubits, config, include_time_step)
    param_values = np.random.uniform(0., np.pi, size=len(trainable_params))
    moment1 = np.zeros_like(param_values)
    moment2 = np.zeros_like(param_values)

    print('  created untrained circuit')

    previous_param_values = param_values.copy()
    for t in range(1, num_epochs + 1):
        penalty_weight = max_penalty_weight * t / num_epochs
        print(f'  Epoch {t} (trash qubit penalty weight: {penalty_weight})')
        param_dict = {param: value for param, value in zip(trainable_params, param_values)}
        encoder_params = {k: v for k,v in param_dict.items() if k in encoder.parameters}
        decoder_params = {k: v for k,v in param_dict.items() if k in decoder.parameters}
        encoder_bound = encoder.assign_parameters(encoder_params)
        decoder_bound = decoder.assign_parameters(decoder_params)

        print('    calculating initial cost')
        # avoid also calculating the loss for (current param - epsilon) and use single
        # initial cost evaluation for all parameters to speed up training
        initial_costs = cost_function(
            training_data, embedder, encoder_bound, decoder_bound, input_params, bottleneck_size, penalty_weight, 1, include_time_step
        )
        cost_history.append(initial_costs)
        initial_cost = initial_costs[0]
        for cost in initial_costs:
            initial_cost += cost
        print('     ', initial_cost)

        gradients = np.zeros_like(param_values)
        # progressively increase the probability that the model will have to deal with it's own noise from the previous time step
        prob_own_noise = 0 # (t-1)/num_epochs
        for j in range(len(param_values)):
            print('    calculating gradient for param ' + str(j+1) + ' / ' + str(len(param_values)) + ' = ' + str((j)/len(param_values)*100) + '% done')
            params_eps = param_values.copy()
            params_eps[j] += gradient_width
            param_dict = {param: value for param, value in zip(trainable_params, params_eps)}
            encoder_params = {k: v for k,v in param_dict.items() if k in encoder.parameters}
            decoder_params = {k: v for k,v in param_dict.items() if k in decoder.parameters}
            encoder_bound = encoder.assign_parameters(encoder_params)
            decoder_bound = decoder.assign_parameters(decoder_params)
            perturbed_costs = cost_function(
                training_data, embedder, encoder_bound, decoder_bound, input_params, bottleneck_size, penalty_weight, 1-prob_own_noise, include_time_step
            )
            perturbed_cost = perturbed_costs[0]
            for cost in perturbed_costs:
                perturbed_cost += cost
            gradients[j] = (perturbed_cost - initial_cost) / gradient_width

        previous_param_values = param_values.copy()
        param_values, moment1, moment2 = adam_update(param_values, gradients, moment1, moment2, t, learning_rate)
        print(f'    Min param update: {np.min(param_values-previous_param_values)}')
        print(f'    Mean param update: {np.mean(param_values-previous_param_values)}')
        print(f'    Std dev param update: {np.std(param_values-previous_param_values)}')
        print(f'    Median param update: {np.median(param_values-previous_param_values)}')
        print(f'    Max param update: {np.max(param_values-previous_param_values)}')

    print('  calculating validation costs')
    param_dict = {param: value for param, value in zip(trainable_params, param_values)}
    encoder_params = {k: v for k,v in param_dict.items() if k in encoder.parameters}
    decoder_params = {k: v for k,v in param_dict.items() if k in decoder.parameters}
    encoder_bound = encoder.assign_parameters(encoder_params)
    decoder_bound = decoder.assign_parameters(decoder_params)
    validation_costs = []
    for (i, series) in validation_data:
        costs = cost_function(
            [(i, series)], embedder, encoder_bound, decoder_bound, input_params, bottleneck_size, penalty_weight, 1, include_time_step
        )
        validation_costs.append([i, costs[0], costs[1]])

    param_dict = {param: value for param, value in zip(trainable_params, param_values)}
    trained_circuit.assign_parameters(param_dict)
    return trained_circuit, cost_history, validation_costs, embedder, encoder_bound, input_params

if __name__ == '__main__':
    from qiskit import qpy
    import os
    import argparse
    import matplotlib.pyplot as plt
    from data_importers import import_generated
    from analysis import entanglement_entropy, von_neumann_entropy

    parser = argparse.ArgumentParser(
        description="Train a Quantum Transition Encoder/Decoder and a Quantum Auto Encoder over each relevant dataset."
    )
    parser.add_argument("data_directory", type=str, help="Path to the directory containing the generated data.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix to use for every saved file name in this run")
    args = parser.parse_args()

    run_prefix = args.prefix if args.prefix else ''
    dataset_partitions = import_generated(args.data_directory)
    num_epochs = 50

    for d_i, (training, validation) in sorted(dataset_partitions.items()):
        num_qubits = len(training[0][1][0])
        bottleneck_size = num_qubits // 2
        config = {
            'bottleneck_size': bottleneck_size,
            'num_blocks': 1,
            'learning_rate': 0.08,
            'max_penalty_weight': 10.0,
            'entanglement_topology': 'full',
            'entanglement_gate': 'cz',
            'embedding_gate': 'rz',
        }
        for model_type in ['qae', 'qae_plus_time', 'qte']:
            np.random.seed(89266583)
            print('Training ' + model_type.upper() + ' for dataset ' + str(d_i))
            include_time_step = model_type == 'qae_plus_time'
            if model_type.startswith('qae'):
                trained_circuit, cost_history, validation_costs, embedder, encoder, input_params = \
                    train_adam(training, validation, qae_cost_function, config, num_epochs=num_epochs, include_time_step=include_time_step)
            elif model_type == 'qte':
                trained_circuit, cost_history, validation_costs, embedder, encoder, input_params = \
                    train_adam(training, validation, qte_cost_function, config, num_epochs=num_epochs)
            else:
                raise Exception('Unknown model_type: ' + model_type)

            print('  Training cost history:', cost_history)
            cost_history = np.array(cost_history)
            fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_cost_history.npy')
            np.save(fname, cost_history)
            print('  Saved cost history')
            print('  Validation cost per series:', validation_costs)
            print(validation_costs)
            validation_costs = np.array(validation_costs)
            fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_validation_costs.npy')
            np.save(fname, validation_costs)
            print('  Saved validation cost per series')

            # === Model metric computations ===
            dataset_enc_entangle_entropies = []
            dataset_enc_vn_entropies = []
            all_trash_indices = []
            for (s_i, series) in validation:
                enc_entangle_entropies = []
                enc_vn_entropies = []
                for t, state in enumerate(series):
                    if include_time_step:
                        params_dict = {p: state[i] for i, p in enumerate(input_params[:-1])}
                        t_param = input_params[-1]
                        params_dict[t_param] = t/len(series)
                    else:
                        params_dict = {p: state[i] for i, p in enumerate(input_params)}
                    qc_init = embedder.assign_parameters(params_dict)
                    initial_dm = DensityMatrix.from_instruction(qc_init)
                    bottleneck_dm = initial_dm.evolve(encoder)

                    # keep track of stats for trash qubit indices histogram
                    marginals = []
                    for q in range(bottleneck_dm.num_qubits):
                        trace_indices = list(range(bottleneck_dm.num_qubits))
                        trace_indices.remove(q)
                        dm_reduced = partial_trace(bottleneck_dm, trace_indices)
                        p0 = np.real(dm_reduced.data[0, 0])
                        marginals.append((q, p0))
                    marginals_sorted = sorted(marginals, key=lambda x: x[1])
                    num_trash = bottleneck_dm.num_qubits - bottleneck_size
                    trash_indices = [q for (q, p) in marginals_sorted[:num_trash]]
                    all_trash_indices.extend(trash_indices)

                    enc_entangle_entropies.append(entanglement_entropy(bottleneck_dm))
                    enc_vn_entropies.append(von_neumann_entropy(bottleneck_dm))
                    if enc_entangle_entropies[-1] > enc_vn_entropies[-1]:
                        difference = enc_entangle_entropies[-1] - enc_vn_entropies[-1]
                        percent = int(100 * difference / enc_vn_entropies[-1])
                        print(f'WARNING: entanglement entropy > full VN entropy for dataset {d_i} series {s_i} state {t} by {difference.real} ({percent}%)')
                dataset_enc_entangle_entropies.append(np.concatenate(([s_i], enc_entangle_entropies)))
                dataset_enc_vn_entropies.append(np.concatenate(([s_i], enc_vn_entropies)))
            plt.figure()
            plt.hist(all_trash_indices, bins=range(0, bottleneck_dm.num_qubits+1), align='left')
            plt.xlabel('Trash Qubit Index')
            plt.ylabel('Frequency')
            plt.title('Histogram of Trash Qubit Selection')
            hist_save_path = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_trash_qubit_histogram.png')
            plt.savefig(hist_save_path)
            print(f'Saved trash qubit histogram to {hist_save_path}')

            def save(dataset_metrics, metric_desc):
                print(f'  {metric_desc}:')
                metric_desc = metric_desc.lower().replace(' ', '_')
                fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_{metric_desc}.npy')
                print('    Shape (series, time steps[+1 for prepending series index]):', dataset_metrics.shape)
                np.save(fname, dataset_metrics)
                print('    Saved', fname)

            dataset_enc_entangle_entropies = np.array(dataset_enc_entangle_entropies)
            save(dataset_enc_entangle_entropies, 'Bottleneck entanglement entropies')

            dataset_enc_vn_entropies = np.array(dataset_enc_vn_entropies)
            save(dataset_enc_vn_entropies, 'Bottleneck full VN entropies')

            fname = os.path.join(args.data_directory, f'{run_prefix}dataset{d_i}_{model_type}_trained_circuit.qpy')
            with open(fname, 'wb') as file:
                qpy.dump(trained_circuit, file)
            print(f"Saved trained circuit to {fname}")
