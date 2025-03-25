from functools import reduce
import numpy as np
import os
import torch
import torch.optim as optim

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace, DensityMatrix, Operator
import re

ENTANGLEMENT_OPTIONS = ['full', 'linear', 'circular']
ENTANGLEMENT_GATES = ['cx', 'cz', 'rzx']

def create_embedding_circuit(num_qubits):
    """
    Returns qc, input_params
    """
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement='full')
    input_params = list(feature_map.parameters)
    return feature_map, input_params

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

def create_qed_circuit(bottleneck_size, num_qubits, num_blocks, entanglement_topology, entanglement_gate):
    """
    Build a parameterized QTE transformation (encoder) using multiple layers.

    For each block, we perform:
      - A layer of single-qubit rotations (we use Ry for simplicity).
      - An entangling layer whose connectivity is determined by entanglement_topology.

    The total number of parameters is:
      num_blocks * (2 * num_qubits)
    (Each layer has 2 * num_qubits parameters: one set before the entangling layer and one set after.)

    The decoder is taken as the inverse of the entire encoder.
    Returns qte_circuit, encoder, decoder_circuit, all_parameters
    """
    encoder = QuantumCircuit(num_qubits)
    params = []
    for layer in range(num_blocks):
        add_entanglement_topology(encoder, num_qubits, entanglement_topology, entanglement_gate)
        # add set of single qubit rotations
        for i in range(num_qubits):
            p = Parameter('Encoder Layer ' + str(layer) + ' Ry θ ' + str(i))
            params.append(p)
            encoder.ry(p, i)

    # For a proper autoencoder, you want to compress information into a bottleneck but
    # choosing which qubits to force to |0> reduces the flexibility the AE has in
    # compressing and reconstructing the data. Instead, in this design, the circuit
    # structure (and subsequent training loss) is expected to force the encoder to
    # focus its information into 'bottleneck_size' qubits.
    decoder = encoder.inverse()
    decoder_params = {
        p: Parameter(p.name.replace("Encoder", "Decoder")) for p in decoder.parameters
    }
    decoder.assign_parameters(decoder_params)
    params.extend(decoder.parameters)
    return encoder.compose(decoder), encoder, decoder, params

def create_full_circuit(num_qubits, config):
    """
    config is dict w/ additional hyperparameters:
        num_blocks:
        bottleneck_size:
        entanglement_topology:   options are ['full', 'linear', 'circular']
        entanglement_gate:       options are ['CX', 'CZ', 'RZX']
    Returns full_circuit, encoder_circuit, decoder_circuit, input_params, trainable_params
    """
    embedding_qc, input_params = create_embedding_circuit(num_qubits)

    num_blocks = config.get('num_blocks', 1)
    ent_topology = config.get('entanglement_topology', 'full')
    ent_gate_type = config.get('entanglement_gate', 'cx')
    bottleneck_size = config.get('bottleneck_size', int(num_qubits/2))
    qte_circuit, encoder, decoder, trainable_params = create_qed_circuit(
        bottleneck_size, num_qubits, num_blocks, ent_topology, ent_gate_type
    )
    return embedding_qc.compose(qte_circuit), embedding_qc, encoder, decoder, input_params, trainable_params

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

def force_trash_qubits(bottleneck_state, bottleneck_size):
    """
    Force the trash qubits in the bottleneck_state to |0> using a density matrix projection.

    This function computes the marginal probability (p0) for each qubit,
    selects the trash qubits dynamically (those with the lowest p0),
    and for each trash qubit applies a projection operator (|0><0| on that qubit)
    to the density matrix of the full state. The resulting density matrix is renormalized
    The use of the marginals to determine the trash qubits instead of the joint pdf is to
    maintain scalability
    """
    marginals = []
    for q in range(bottleneck_state.num_qubits):
        trace_indices = list(range(bottleneck_state.num_qubits))
        trace_indices.remove(q)
        dm = partial_trace(bottleneck_state, trace_indices)
        p0 = np.real(dm.data[0, 0])
        marginals.append((q, p0))

    # automatically select trash qubits as the ones with lowest marginals
    sorted_marginals = sorted(marginals, key=lambda x: x[1])
    num_trash = bottleneck_state.num_qubits - bottleneck_size
    trash_qubit_indices = [q for (q, p0) in sorted_marginals[:num_trash]]

    dm_full = DensityMatrix(bottleneck_state)

    # For each trash qubit, apply the projection operator onto |0>
    # Note: Qiskit uses little-endian ordering for statevectors. That is,
    # the rightmost bit in the binary representation corresponds to qubit 0.
    # When constructing the full tensor-product projector, we prepare a list
    # of 2×2 operators with P0 inserted at position (n-1 - q).
    P0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    for q in trash_qubit_indices:
        ops = []
        # build a list of operators for each qubit
        for i in range(bottleneck_state.num_qubits):
            if i == (bottleneck_state.num_qubits - 1 - q):
                ops.append(P0)
            else:
                ops.append(I)
        proj = reduce(np.kron, ops)
        dm_full = DensityMatrix(proj @ dm_full.data @ proj.conj().T)
        # renormalize to ensure a valid quantum state
        dm_full = dm_full / dm_full.trace()

    return dm_full

def dm_to_statevector(dm):
    """
    Convert a (nearly pure) DensityMatrix dm to a Statevector.
    This function computes the eigen-decomposition of dm.data and returns
    the eigenvector corresponding to the largest eigenvalue.
    """
    evals, evecs = np.linalg.eig(dm.data)
    idx = np.argmax(evals)
    vec = evecs[:, idx]
    vec = vec / np.linalg.norm(vec)
    return Statevector(vec)


PAULI_MATRICES = {
    'rx': np.array([[0, 1], [1, 0]], dtype=complex),
    'ry': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'rz': np.array([[1, 0], [0, -1]], dtype=complex)
}

def expand_operator(single_op, target, total_qubits):
    """
    Expand a single-qubit operator to act on a multi-qubit Hilbert space.

    Args:
        single_op (np.ndarray): 2x2 operator.
        target (int): the qubit index the operator acts on.
        total_qubits (int): total number of qubits.

    Returns:
        np.ndarray: Full operator of shape (2**total_qubits, 2**total_qubits).
    """
    op = 1
    for i in range(total_qubits):
        if i == target:
            op = np.kron(op, single_op)
        else:
            op = np.kron(op, np.eye(2, dtype=complex))
    return op

def adam_update(params, gradients, moment1, moment2, t, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    moment1 = beta1 * moment1 + (1 - beta1) * gradients
    moment2 = beta2 * moment2 + (1 - beta2) * (gradients ** 2)
    bias_corrected_moment1 = moment1 / (1 - beta1 ** t)
    bias_corrected_moment2 = moment2 / (1 - beta2 ** t)
    new_params = params - lr * bias_corrected_moment1 / (np.sqrt(bias_corrected_moment2) + epsilon)
    return new_params, moment1, moment2

# consider DARBO (https://arxiv.org/pdf/2303.14877) in the future due to high complexity and high nonstationarity of loss landscape
def train_adam(training_data, validation_data, model_type, config, num_epochs=100):
    """
    Train the QTE (or QAE) by minimizing the cost function using ADAM with gradients
    computed via adjoint differentiation computed layer-by-layer (i.e., per block) like backprop.

    If config['model_type'] is 'qae', then each state is processed individually,
    with target_state = current_state (i.e. reconstruct the same state).
    If 'qte', then target_state is the next state in the series.
    """
    num_qubits = len(training_data[0][1][0])
    num_params = 4 * num_qubits * config['num_blocks']  # 2 layers per block
    param_values = np.random.uniform(0., np.pi, size=num_params)
    cost_history = []
    moment1 = np.zeros_like(param_values)
    moment2 = np.zeros_like(param_values)

    bottleneck_size = int(config['bottleneck_size'])
    num_blocks = int(config['num_blocks'])
    learning_rate = float(config['learning_rate'])
    penalty_weight = float(config.get('penalty_weight', 1.0))
    teacher_forcing_prob = config.get('teacher_forcing_prob', 1.0)
    model_type = config.get('model_type', 'qte')  # 'qae' or 'qte'

    trained_circuit, embedder, encoder, decoder, input_params, trainable_params = create_full_circuit(num_qubits, config)
    print('  created untrained circuit')

    # The circuit block used for gradient computation (encoder followed by decoder)
    circuit_block = encoder.compose(decoder)

    def adjoint_gradients_layerwise(cost_function, circuit, pvals, parameters, num_blocks, num_qubits, target_state):
        """
        Compute gradients for a given circuit block (encoder composed with decoder) in a layer-wise (per-block)
        fashion, similar to backpropagation.

        Parameters:
          cost_function: function mapping a final state to a cost (assumed to be C = 1 - |<target|state>|^2)
          circuit: a QuantumCircuit representing the block (encoder.compose(decoder))
          pvals: current parameter values (numpy array)
          parameters: list of trainable Parameter objects (in order) corresponding to pvals
          num_blocks: number of blocks/layers in the circuit
          num_qubits: number of qubits in the circuit
          target_state: a Statevector for the desired target state

        Returns:
          gradients: numpy array of gradients for each parameter.

        The analytic adjoint state is derived as follows:
          Let f = <target|state>, then for cost C = 1 - |f|^2, we have
             dC/d(state*) = - f * target.
        """
        total_qubits = circuit.num_qubits
        params_per_block = 2 * num_qubits  # parameters per block

        bound_circuit = circuit.assign_parameters({param: value for param, value in zip(parameters, pvals)})

        forward_states = []   # will store state at block boundaries
        block_gate_infos = [] # will store lists of (instruction, qargs, parameter index, parameter value) per block
        current_block = []
        param_counter = 0

        # Forward pass: iterate over the already-bound circuit's instructions.
        # Use the new API: each item in bound_circuit.data is a CircuitInstruction.
        state = Statevector.from_label("0" * total_qubits)
        for item in bound_circuit.data:
            instr = item.operation
            qargs = item.qubits
            # If this instruction was parameterized originally, record it.
            if instr.params and len(instr.params) > 0:
                current_block.append((instr, qargs, param_counter, pvals[param_counter]))
                param_counter += 1
                if param_counter % params_per_block == 0:
                    forward_states.append(state.data.copy())
                    block_gate_infos.append(current_block)
                    current_block = []
            U = Operator(instr).data
            state = state.evolve(U)

        # Derive the analytic adjoint state.
        # For cost C = 1 - |<target|state>|^2, let f = <target|state>, then:
        #    dC/d(state*) = - f * target.
        f = np.vdot(target_state.data, state.data)
        adj_state = - f * target_state.data
        norm_adj = np.linalg.norm(adj_state)
        if norm_adj != 0:
            adj_state = adj_state / norm_adj
        else:
            adj_state = np.ones_like(state.data, dtype=complex) / np.linalg.norm(np.ones_like(state.data, dtype=complex))

        gradients = np.zeros_like(pvals)

        # Backward pass: iterate over blocks in reverse order.
        for b in reversed(range(len(block_gate_infos))):
            for instr, qargs, idx, param_val in reversed(block_gate_infos[b]):
                state_before = forward_states[b]  # state recorded at block boundary
                # Here, instr is already bound.
                U = Operator(instr).data
                gate_name = instr.name.lower()
                if gate_name in ['rx', 'ry', 'rz']:
                    P = PAULI_MATRICES[gate_name]
                    dU = (-1j/2) * np.dot(P, U)
                    # Expand derivative operator to full Hilbert space.
                    target_index = qargs[0].index if hasattr(qargs[0], "index") else qargs[0]
                    D_full = expand_operator(dU, target_index, total_qubits)
                    grad_contrib = np.real(np.vdot(adj_state, np.dot(D_full, state_before)))
                    gradients[idx] = grad_contrib
                else:
                    gradients[idx] = 0
                # Update the adjoint state.
                target_index = qargs[0].index if hasattr(qargs[0], "index") else qargs[0]
                U_full = expand_operator(U, target_index, total_qubits)
                adj_state = np.dot(np.conjugate(U_full.T), adj_state)
        return gradients

    # Training loop:
    total_gradients = np.zeros_like(param_values)
    total_transition_cost = 0.0
    total_transitions = 0

    for t in range(1, num_epochs + 1):
        print('  Epoch', t)
        epoch_gradients = np.zeros_like(param_values)
        epoch_cost = 0.0
        count = 0

        # Loop over each series in training data.
        for (_, series) in training_data:
            series_length = len(series)
            if series_length < 1:
                continue

            if model_type.lower() == 'qae':
                for state in series:
                    params = {p: state[i] for i, p in enumerate(input_params)}
                    qc_state = embedder.assign_parameters(params)
                    current_state = Statevector.from_instruction(qc_state)
                    target_state = current_state
                    def cost_state(pvals):
                        param_dict = {param: value for param, value in zip(trainable_params, pvals)}
                        encoder_bound = encoder.assign_parameters(param_dict)
                        decoder_bound = decoder.assign_parameters(param_dict)
                        predicted_state = current_state.evolve(encoder_bound).evolve(decoder_bound)
                        return 1 - state_fidelity(current_state, predicted_state)
                    epoch_gradients += adjoint_gradients_layerwise(
                        cost_state, circuit_block, param_values, trainable_params, config['num_blocks'], num_qubits, target_state
                    )
                    epoch_cost += cost_state(param_values)
                    count += 1
            elif model_type.lower() == 'qte':
                # For QTE, process transitions.
                # Always use ground truth for the first time step.
                params = {p: series[0][i] for i, p in enumerate(input_params)}
                qc_initial = embedder.assign_parameters(params)
                current_state = Statevector.from_instruction(qc_initial)
                for i in range(series_length - 1):
                    next_state = series[i+1]
                    next_params = {p: next_state[j] for j, p in enumerate(input_params)}
                    qc_next = embedder.assign_parameters(next_params)
                    target_state = Statevector.from_instruction(qc_next)
                    def cost_trans(pvals):
                        param_dict = {param: value for param, value in zip(trainable_params, pvals)}
                        encoder_bound = encoder.assign_parameters(param_dict)
                        decoder_bound = decoder.assign_parameters(param_dict)
                        predicted_state = current_state.evolve(encoder_bound).evolve(decoder_bound)
                        return 1 - state_fidelity(target_state, predicted_state)
                    epoch_gradients += adjoint_gradients_layerwise(
                        cost_trans, circuit_block, param_values, trainable_params, config['num_blocks'], num_qubits, target_state
                    )
                    epoch_cost += cost_trans(param_values)
                    count += 1
                    # Teacher forcing update.
                    if np.random.rand() < teacher_forcing_prob:
                        current_state = target_state
                    else:
                        encoder_bound = encoder.assign_parameters({p: v for p, v in zip(trainable_params, param_values)})
                        decoder_bound = decoder.assign_parameters({p: v for p, v in zip(trainable_params, param_values)})
                        current_state = current_state.evolve(encoder_bound).evolve(decoder_bound)
                else:
                    raise Exception('Unknown model_type: ' + model_type)
        if count > 0:
            avg_gradients = epoch_gradients / count
            avg_cost = epoch_cost / count
        else:
            avg_gradients = epoch_gradients
            avg_cost = 0
        cost_history.append(avg_cost)
        print("    Avg cost:", avg_cost)
        prev_params = param_values.copy()
        param_values, moment1, moment2 = adam_update(param_values, avg_gradients, moment1, moment2, t, learning_rate)
        print("    Mean param update:", np.mean(param_values - prev_params))

    print("  calculating validation cost")
    val_cost = 0.0
    transitions = 0
    for (_, series) in validation_data:
        series_length = len(series)
        if series_length < 1:
            continue
        if model_type == 'qae':
            for state in series:
                params = {p: state[i] for i, p in enumerate(input_params)}
                qc_state = embedder.assign_parameters(params)
                current_state = Statevector.from_instruction(qc_state)
                encoder_bound = encoder.assign_parameters({p: v for p, v in zip(trainable_params, param_values)})
                decoder_bound = decoder.assign_parameters({p: v for p, v in zip(trainable_params, param_values)})
                predicted_state = current_state.evolve(encoder_bound).evolve(decoder_bound)
                val_cost += 1 - state_fidelity(current_state, predicted_state)
                transitions += 1
        else:
            params = {p: series[0][i] for i, p in enumerate(input_params)}
            qc_initial = embedder.assign_parameters(params)
            current_state = Statevector.from_instruction(qc_initial)
            for i in range(series_length - 1):
                next_state = series[i+1]
                next_params = {p: next_state[j] for j, p in enumerate(input_params)}
                qc_next = embedder.assign_parameters(next_params)
                target_state = Statevector.from_instruction(qc_next)
                encoder_bound = encoder.assign_parameters({p: v for p, v in zip(trainable_params, param_values)})
                decoder_bound = decoder.assign_parameters({p: v for p, v in zip(trainable_params, param_values)})
                predicted_state = current_state.evolve(encoder_bound).evolve(decoder_bound)
                val_cost += 1 - state_fidelity(target_state, predicted_state)
                transitions += 1
                if np.random.rand() < teacher_forcing_prob:
                    current_state = target_state
                else:
                    current_state = predicted_state
    validation_cost = val_cost / transitions
    print('  Validation Cost:', validation_cost)
    param_dict = {param: value for param, value in zip(trainable_params, param_values)}
    trained_circuit.assign_parameters(param_dict)
    return trained_circuit, cost_history, validation_cost, embedder, encoder, input_params

if __name__ == '__main__':
    from qiskit import qpy
    import os
    import argparse
    from data_importers import import_generated
    from analysis import differential_entropy, entanglement_entropy, entropy

    parser = argparse.ArgumentParser(
        description="Train a Quantum Transition Encoder/Decoder and a Quantum Auto Encoder over the given data."
    )
    parser.add_argument("data_directory", type=str, help="Path to the directory containing the generated data.")
    args = parser.parse_args()

    dataset_partitions = import_generated(args.data_directory)
    num_epochs = 50

    for d_i, (training, validation) in sorted(dataset_partitions.items()):
        num_qubits = len(training[0][1][0])
        bottleneck_size = num_qubits // 2
        config = {
            'bottleneck_size': bottleneck_size,
            'num_blocks': 1,
            'learning_rate': .1,
            'penalty_weight': .75,
            'entanglement_topology': 'full',
            'entanglement_gate': 'rzx',
            'embedding_gate': 'rz',
        }
        for model_type in ['qae', 'qte']:
            print('Training ' + model_type.upper() + ' for dataset ' + str(d_i))
            trained_circuit, cost_history, validation_cost, embedder, encoder, input_params = \
                train_adam(training, validation, model_type, config, num_epochs=num_epochs)

            print('  Final training cost:', cost_history[-1])
            print('  Validation cost:', validation_cost)

            # === Entropy computations ===
            dataset_bottleneck_entropies = []
            for (_, series) in validation:
                series_entropies = []  # list to hold entropy values for each series in this dataset
                for state_series in series:  # assume each "state_series" is a time series (list of states)
                    time_step_entropies = []  # entropy at each time step in this series
                    for state in state_series:
                        params_dict = {p: state[i] for i, p in enumerate(input_params)}
                        qc_init = embedder.assign_parameters(params_dict)
                        initial_dm = DensityMatrix.from_instruction(qc_init)
                        encoder_dm = initial_dm.evolve(encoder)

                        # Compute marginals for each qubit (here using probability of |0⟩ from the reduced density matrix)
                        marginals = []
                        for q in range(encoder_dm.num_qubits):
                            trace_indices = list(range(encoder_dm.num_qubits))
                            trace_indices.remove(q)
                            reduced_dm = partial_trace(encoder_dm, trace_indices)
                            marginals.append((q, np.real(reduced_dm.data[0, 0])))

                        # Sort qubits by the probability of |0⟩.
                        sorted_marginals = sorted(marginals, key=lambda x: x[1])
                        num_trash = encoder_dm.num_qubits - bottleneck_size
                        # Keep the qubits with higher |0⟩ probability (i.e. those not in the trash).
                        bottleneck_indices = sorted([q for q, _ in sorted_marginals[num_trash:]])

                        keep_set = set(bottleneck_indices)
                        trace_out = [i for i in range(encoder_dm.num_qubits) if i not in keep_set]
                        bottleneck_dm = partial_trace(encoder_dm, trace_out)

                        time_step_entropies.append(entropy(bottleneck_dm))
                    series_entropies.append(np.array(time_step_entropies))
                dataset_bottleneck_entropies.append(np.array(series_entropies))

            dataset_bottleneck_entropies = np.array(dataset_bottleneck_entropies)
            print('  Bottleneck VN entropy shape (datasets, series, time steps):', dataset_bottleneck_entropies.shape)
            fname = os.path.join(args.data_directory, f'dataset{d_i}_{model_type}_bottleneck_vn_entropies.npy')
            np.save(fname, dataset_bottleneck_entropies)
            print(f'  Saved bottleneck VN entropy calculations to {fname}')
            avg_entropy_per_series = np.mean(dataset_bottleneck_entropies, axis=2)
            print('Average bottleneck entropy per series:', avg_entropy_per_series)
            avg_entropy_per_dataset = np.mean(avg_entropy_per_series, axis=1)
            print('Average bottleneck entropy per dataset:', avg_entropy_per_dataset)
            overall_avg_entropy = np.mean(avg_entropy_per_dataset)
            print(f'Overall average bottleneck entropy: {overall_avg_entropy}')

            # encoder entanglement entropy
            entanglement_entropies = []
            subsystem = list(range(bottleneck_size))
            for series in validation:
                series_entanglement_entropy = []
                for state in series:
                    params_dict = {p: state[i] for i, p in enumerate(input_params)}
                    qc_init = embedder.assign_parameters(params_dict)
                    initial_state = Statevector.from_instruction(qc_init)
                    encoder_state = initial_state.evolve(encoder)
                    ent_entropy = entanglement_entropy(encoder_state, subsystem)
                    series_entanglement_entropy.append(ent_entropy)
                entanglement_entropies.append(series_entanglement_entropy)
            entanglement_entropies = np.array(entanglement_entropies)
            fname = os.path.join(args.data_directory, f'dataset{d_i}_{model_type}_encoder_entanglement_entropy.npy')
            np.save(fname, entanglement_entropies)
            avg_entanglement_entropy = np.mean(entanglement_entropies)
            print(f'  Average encoder entanglement entropy: {avg_entanglement_entropy:.6f}')

            from qiskit import qpy
            fname = os.path.join(args.data_directory, f'dataset{d_i}_{type}_trained_circuit.qpy')
            with open(fname, 'wb') as file:
                qpy.dump(trained_circuit, file)
            print(f"Saved trained circuit to {fname}")
