import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix

from utility import dm_to_statevector, without_t_gate, fix_dm_array, normalize_classical_vector

ENTANGLEMENT_OPTIONS = ['full', 'linear', 'circular']
ENTANGLEMENT_GATES = ['cx', 'cz', 'rzx']
ROTATION_GATES = ['rx', 'ry', 'rz']

class QuantumEncoderDecoder:
    hidden_state_weight_param = Parameter('Hidden State Weight')

    def __init__(self, num_qubits, config, is_recurrent=False):
        self.num_qubits = num_qubits
        self.num_blocks = config.get('num_blocks', 1)
        self.entanglement_topology = config.get('entanglement_topology', 'full')
        self.entanglement_gate = config.get('entanglement_gate', 'cx')
        self.block_gate = config.get('block_gate', 'ry')
        self.embedding_gate = config.get('embedding_gate', 'rz')
        self.bottleneck_size = config.get('bottleneck_size', num_qubits//2)
        self.is_recurrent = is_recurrent

        self.create_embedding_circuit()
        self.full_circuit = self.embedder.compose(self.create_qed_circuit())
        self.hidden_state = None
        self.hidden_weight = None

    @property
    def num_features(self):
        return self.num_qubits

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, state):
        if not isinstance(state, DensityMatrix):
            state = DensityMatrix(state)
        bottleneck_dm = state.evolve(self.encoder_bound)
        if self.is_recurrent:
            if self.hidden_state is None:
                self.hidden_state = np.zeros_like(bottleneck_dm.data)
            # force weight between 0 and 1 w/o creating flat part of loss landscape
            weight = 1.0 / (1.0 + np.exp(-self.hidden_weight))
            self.hidden_state = (1-weight)*bottleneck_dm.data + weight*self.hidden_state
            bottleneck_dm = DensityMatrix(self.hidden_state)

        predicted_state = bottleneck_dm.evolve(self.decoder_bound)
        if predicted_state.data.ndim > 1:
            predicted_state = dm_to_statevector(predicted_state)
        return bottleneck_dm, predicted_state

    def prepare_state(self, state):
        param_values = {p: state[i] for i, p in enumerate(self.input_params)}
        return Statevector.from_instruction(self.embedder.assign_parameters(param_values))

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
        self.embedder = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            self.input_params.append(self.add_rotation_gate(self.embedder, self.embedding_gate, 'Embedding Rθ ' + str(i), i))

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
        self.encoder = QuantumCircuit(self.num_qubits)
        self.trainable_params = []
        if self.is_recurrent:
            self.trainable_params.append(self.hidden_state_weight_param)
        for layer in range(self.num_blocks):
            self.add_entanglement_topology(self.encoder)
            for i in range(self.num_qubits):
                self.add_rotation_gate(self.encoder, self.block_gate, 'Encoder Layer ' + str(layer) + ' Ry θ ' + str(i), i)
        self.trainable_params.extend(self.encoder.parameters)

        # For a proper autoencoder, you want to compress information into a bottleneck but
        # choosing which qubits to force to |0> reduces the flexibility the AE has in
        # compressing and reconstructing the data. Instead, in this design, the circuit
        # structure (and subsequent training loss) is expected to force the encoder to
        # focus its information into 'bottleneck_size' qubits. This helps mitigate mode collapse.
        self.decoder = QuantumCircuit(self.num_qubits)
        for layer in range(self.num_blocks):
            self.add_entanglement_topology(self.decoder)
            for i in range(self.num_qubits):
                self.add_rotation_gate(self.decoder, self.block_gate, 'Decoder Layer ' + str(layer) + ' Ry θ ' + str(i), i)
        self.trainable_params.extend(self.decoder.parameters)
        return self.encoder.compose(self.decoder)

    def add_rotation_gate(self, circuit, gate, description, qubit_index):
        p = Parameter(f'{description}')
        if gate.lower() == 'rx':
            circuit.rx(p, qubit_index)
        elif gate.lower() == 'ry':
            circuit.ry(p, qubit_index)
        elif gate.lower() == 'rz':
            circuit.rz(p, qubit_index)
        else:
            raise Exception("Invalid rotation gate: " + gate)
        return p

    def set_params(self, params_dict):
        encoder_params = {k: v for k,v in params_dict.items() if k in self.encoder.parameters}
        decoder_params = {k: v for k,v in params_dict.items() if k in self.decoder.parameters}
        self.encoder_bound = self.encoder.assign_parameters(encoder_params)
        self.decoder_bound = self.decoder.assign_parameters(decoder_params)
        if self.hidden_state_weight_param in params_dict:
            if self.hidden_weight is None:
                # always start at zero to give better starting gradient, ensuring contribution
                # from both hidden and current bottleneck
                self.hidden_weight = 0
            else:
                self.hidden_weight = params_dict[self.hidden_state_weight_param]


class RestrictedParamCountIndividualCayleyLinear(nn.Module):
    """
    A linear layer with a specified even number of free parameters equal to both the input and output dimension.
    To achieve this, they are split in half between an two smaller matrices, each with half dimensions. One is
    an orthogonal rotation matrix created via a Cayley transform and the other is a diagonal scaling matrix. The
    core transformation is a scaled rotation. The dimensionality of this core transformation is then increased by
    repeatedly adding (with overlap) the lower-dimensional result along the diagonal of the higher dimensional
    matrix so that each individual node has a unique rotation while still forcing the free parameters to be
    "coupled" across the layer. While this is not a perfect analogue to the quantum architecture — since in a
    quantum system the qubits themselves are inherently correlated — it does allow a correlation between the
    effects of the rotations. This engineered coupling mimics, to some extent, the way local gate parameters
    interact in quantum circuits, though it does not reproduce the full complexity of quantum entanglement.
    """
    def __init__(self, num_params):
        super(RestrictedParamCountIndividualCayleyLinear, self).__init__()
        if num_params % 2 == 1:
            raise Exception('Need even number of features')
        self.num_params = num_params
        self.rotation_params = nn.Parameter(torch.randn(self.num_params // 2))
        self.diagonal_scaling_params = nn.Parameter(torch.randn(self.num_params // 2))

    def forward(self, x):
        skew_symmetric_matrix = self.rotation_params.unsqueeze(1) - self.rotation_params.unsqueeze(0)
        I = torch.eye(self.num_params // 2, device=x.device, dtype=x.dtype)
        rotation = torch.linalg.solve(I - skew_symmetric_matrix, I + skew_symmetric_matrix)
        scaling = torch.diag(self.diagonal_scaling_params)
        core = scaling @ rotation
        lifted_core = torch.zeros(self.num_params, self.num_params, dtype=core.dtype)
        for offset in range(self.num_params//2 + 1):
            rotated_core = torch.rot90(core, k=offset, dims=(0, 1)) # to get a different rotation per feature
            lifted_core[offset:offset+self.num_params // 2, offset:offset+self.num_params // 2] += rotated_core
        return x @ lifted_core.T


class ClassicalEncoderDecoder(nn.Module):
    def __init__(self, num_features, config, is_recurrent=False):
        super(ClassicalEncoderDecoder, self).__init__()
        self.num_features = num_features
        self.is_recurrent = is_recurrent
        self.num_blocks = config.get('num_blocks', 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.encoder.append(RestrictedParamCountIndividualCayleyLinear(num_features))
            # bottleneck enforced via cost function similar to Quantum version
            self.decoder.append(RestrictedParamCountIndividualCayleyLinear(num_features))
        self.bottleneck_size = config.get('bottleneck_size', self.num_features//2)
        if self.is_recurrent:
            # always start at zero to give better starting gradient
            self.hidden_weight = nn.Parameter(torch.tensor([0.0]))
        self.hidden_state = None
        # ensure set_params starts w/ 0 for hidden weight
        self._params_initialized = False

    @property
    def trainable_params(self):
        all_params = []
        for p_tensor in self.parameters():
            for p in p_tensor.flatten():
                all_params.append(p)
        return all_params

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(self.num_features)
        bottleneck_state = x
        for block in self.encoder:
            bottleneck_state = block(bottleneck_state)

        if self.is_recurrent:
            # add normalization to avoid infinite growth and have similar dynamics as
            # fixing of density matrix in quantum version
            # force weight between 0 and 1 w/o creating flat part of loss landscape
            weight = 1.0 / (1.0 + np.exp(-self.hidden_weight.detach().numpy()[0]))
            bottleneck_state = (1-weight)*bottleneck_state + weight*self.hidden_state
            output = self.hidden_state = bottleneck_state
        else:
            output = bottleneck_state

        for block in self.decoder:
            output = block(output)
        return bottleneck_state.detach().numpy(), output.detach().numpy()

    def prepare_state(self, state):
        return torch.Tensor(state)

    def get_trash_indices(self, bottleneck_state):
        num_trash = bottleneck_state.shape[0] - self.bottleneck_size
        indices = []
        for s in range(len(bottleneck_state)):
            # use the lowest magnitude features as trash
            indices.append((s, abs(bottleneck_state[s])))
        return [i for (i, v) in sorted(indices, key=lambda x: x[1])[:num_trash]]

    def set_params(self, params_dict):
        for p, v in params_dict.items():
            if not self._params_initialized and self.is_recurrent and p == self.hidden_weight:
                self._params_initialized = True
                continue # skip first assignment to always start w/ 0
            with torch.no_grad():
                p.copy_(torch.tensor(v, dtype=torch.float32))
