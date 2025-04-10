import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import partial_trace, Statevector

from utility import dm_to_statevector, without_t_gate

ENTANGLEMENT_OPTIONS = ['full', 'linear', 'circular']
ENTANGLEMENT_GATES = ['cx', 'cz', 'rzx']

class QuantumEncoderDecoder:
    def __init__(self, num_qubits, config, include_time_step=False):
        self.num_qubits = num_qubits
        self.num_blocks = config.get('num_blocks', 1)
        self.entanglement_topology = config.get('entanglement_topology', 'full')
        self.entanglement_gate = config.get('entanglement_gate', 'cx')
        self.embedding_gate = config.get('embedding_gate', 'rz')
        self.bottleneck_size = config.get('bottleneck_size', num_qubits//2)
        self.include_time_step = include_time_step

        self.create_embedding_circuit()
        self.full_circuit = self.embedder.compose(self.create_qed_circuit())

    @property
    def num_features(self):
        if self.include_time_step:
            return self.num_qubits + 1
        return self.num_qubits

    def forward(self, state):
        bottleneck_state = state.evolve(self.encoder_bound)
        predicted_state = bottleneck_state.evolve(self.decoder_bound)
        if self.include_time_step:
            predicted_state = partial_trace(predicted_state, [self.num_qubits-1])
        if predicted_state.data.ndim > 1:
            predicted_state = dm_to_statevector(predicted_state)
        return bottleneck_state, predicted_state

    def prepare_state(self, state, t_value=None, is_output_state=False):
        embedder = self.embedder
        if self.include_time_step:
            param_values = {p: state[i] for i, p in enumerate(self.input_params[:-1])} # final param is time step
            if not is_output_state:
                if t_value is None:
                    raise Exception('Missing expected time step feature value')
                t_param = self.input_params[-1]
                param_values[t_param] = t_value
            else:
                embedder = without_t_gate(embedder)
        else:
            param_values = {p: state[i] for i, p in enumerate(self.input_params)}
        return Statevector.from_instruction(embedder.assign_parameters(param_values))

    def get_trash_indices(self, bottleneck_dm):
        num_trash = bottleneck_dm.num_qubits - self.bottleneck_size
        marginals = []
        for q in range(bottleneck_dm.num_qubits):
            trace_indices = list(range(bottleneck_dm.num_qubits))
            trace_indices.remove(q)
            dm_reduced = partial_trace(bottleneck_dm, trace_indices)
            p0 = np.real(dm_reduced.data[0, 0])
            marginals.append((q, p0))
        return [q for (q, p) in sorted(marginals, key=lambda x: x[1])[:num_trash]]

    def create_embedding_circuit(self):
        """
        Apply rotation gate to each qubit for embedding of classical data.
        """
        self.input_params = []
        n_qbits = self.num_qubits+1 if self.include_time_step else self.num_qubits
        self.embedder = QuantumCircuit(n_qbits)
        for i in range(n_qbits):
            if self.include_time_step and i == n_qbits-1:
                p = Parameter('t')
            else:
                p = Parameter('Embedding Rθ ' + str(i))
            self.input_params.append(p)
            if self.embedding_gate.lower() == 'rx':
                self.embedder.rx(p, i)
            elif self.embedding_gate.lower() == 'ry':
                self.embedder.ry(p, i)
            elif self.embedding_gate.lower() == 'rz':
                self.embedder.rz(p, i)
            else:
                raise Exception("Invalid embedding gate: " + self.embedding_gate)

    def add_entanglement_topology(self, qc: QuantumCircuit):
        if self.entanglement_topology == 'full':
            for i in range(self.num_qubits):
                for j in range(i+1, self.num_qubits):
                    if self.entanglement_gate.lower() == 'cx':
                        qc.cx(i, j)
                    elif self.entanglement_gate.lower() == 'cz':
                        qc.cz(i, j)
                    elif self.entanglement_gate.lower() == 'rzx':
                        qc.rzx(np.pi/4, i, j)
                    else:
                        raise Exception("Unknown entanglement gate: " + self.entanglement_gate)
        elif self.entanglement_topology == 'linear':
            for i in range(self.num_qubits - 1):
                if self.entanglement_gate.lower() == 'cx':
                    qc.cx(i, i+1)
                elif self.entanglement_gate.lower() == 'cz':
                    qc.cz(i, i+1)
                elif self.entanglement_gate.lower() == 'rzx':
                    qc.rzx(np.pi/4, i, i+1)
                else:
                    raise Exception("Unknown entanglement gate: " + self.entanglement_gate)
        elif self.entanglement_topology == 'circular':
            for i in range(self.num_qubits - 1):
                if self.entanglement_gate.lower() == 'cx':
                    qc.cx(i, i+1)
                elif self.entanglement_gate.lower() == 'cz':
                    qc.cz(i, i+1)
                elif self.entanglement_gate.lower() == 'rzx':
                    qc.rzx(np.pi/4, i, i+1)
                else:
                    raise Exception("Unknown entanglement gate: " + self.entanglement_gate)
            if self.entanglement_gate.lower() == 'cx':
                qc.cx(self.num_qubits-1, 0)
            elif self.entanglement_gate.lower() == 'cz':
                qc.cz(self.num_qubits-1, 0)
            elif self.entanglement_gate.lower() == 'rzx':
                qc.rzx(np.pi/4, self.num_qubits-1, 0)
            else:
                raise Exception("Unknown entanglement gate: " + self.entanglement_gate)

    def create_qed_circuit(self):
        """
        Build a parameterized encoder using multiple layers.

        For each block, we perform:
          - A layer of single-qubit rotations (we use Ry for simplicity).
          - An entangling layer whose connectivity is determined by entanglement_topology.
        """
        n_qbits = self.num_qubits
        if self.include_time_step:
            n_qbits += 1
        self.encoder = QuantumCircuit(n_qbits)
        self.trainable_params = []
        for layer in range(self.num_blocks):
            self.add_entanglement_topology(self.encoder)
            for i in range(n_qbits):
                self.encoder.ry(Parameter('Encoder Layer ' + str(layer) + ' Ry θ ' + str(i)), i)
        self.trainable_params.extend(self.encoder.parameters)

        # For a proper autoencoder, you want to compress information into a bottleneck but
        # choosing which qubits to force to |0> reduces the flexibility the AE has in
        # compressing and reconstructing the data. Instead, in this design, the circuit
        # structure (and subsequent training loss) is expected to force the encoder to
        # focus its information into 'bottleneck_size' qubits. This helps mitigate mode collapse.
        self.decoder = QuantumCircuit(n_qbits)
        # leave the time step qubit in the circuit but don't add any gates to it
        if self.include_time_step:
            n_qbits -= 1
        for layer in range(self.num_blocks):
            self.add_entanglement_topology(self.decoder)
            for i in range(n_qbits):
                self.decoder.ry(Parameter('Decoder Layer ' + str(layer) + ' Ry θ ' + str(i)), i)
        self.trainable_params.extend(self.decoder.parameters)
        return self.encoder.compose(self.decoder)

    def set_params(self, params_dict):
        encoder_params = {k: v for k,v in params_dict.items() if k in self.encoder.parameters}
        decoder_params = {k: v for k,v in params_dict.items() if k in self.decoder.parameters}
        self.encoder_bound = self.encoder.assign_parameters(encoder_params)
        self.decoder_bound = self.decoder.assign_parameters(decoder_params)


class ClassicalEncoderDecoder(nn.Module):
    def __init__(self, num_dimensions, config, include_time_step=False):
        super(ClassicalEncoderDecoder, self).__init__()
        self.num_dimensions = num_dimensions
        self.include_time_step = include_time_step
        if include_time_step:
            num_dimensions += 1
        self.num_blocks = config.get('num_blocks', 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(self.num_blocks):
            # TODO need method for limiting representation capacity for fairness (i.e. only unitary transforms in each layer)
            self.encoder.append(nn.Linear(num_dimensions, num_dimensions))
            # bottleneck enforced via cost function similar to Quantum version to help mitigate mode collapse
            if include_time_step:
                self.decoder.append(nn.Linear(num_dimensions, num_dimensions-1))
            else:
                self.decoder.append(nn.Linear(num_dimensions, num_dimensions))
        self.bottleneck_size = config.get('bottleneck_size', self.num_dimensions//2)

    @property
    def num_features(self):
        if self.include_time_step:
            return self.num_dimensions + 1
        return self.num_dimensions

    @property
    def trainable_params(self):
        all_params = []
        for p_tensor in self.parameters():
            for p in p_tensor.flatten():
                all_params.append(p)
        return all_params

    def forward(self, x):
        bottleneck = torch.Tensor(x)
        for block in self.encoder:
            bottleneck = block(bottleneck)
        output = bottleneck
        for block in self.decoder:
            output = block(output)
        return bottleneck.detach().numpy(), output.detach().numpy()

    def prepare_state(self, state, t_value=None, is_output_state=False):
        if self.include_time_step and not is_output_state:
            if t_value is None:
                raise Exception('Missing expected time step feature value')
            return np.concatenate((state, [t_value]))
        return state

    def get_trash_indices(self, bottleneck_state):
        num_trash = bottleneck_state.shape[0] - self.bottleneck_size
        indices = []
        for s in range(len(bottleneck_state)):
            # use the lowest magnitude features as trash
            indices.append((s, abs(bottleneck_state[s])))
        return [i for (i, v) in sorted(indices, key=lambda x: x[1])[:num_trash]]

    def set_params(self, params_dict):
        for p, v in params_dict.items():
            with torch.no_grad():
                p.copy_(torch.tensor(v, dtype=torch.float32))
