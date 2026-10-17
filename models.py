import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix, Pauli

ENTANGLEMENT_OPTIONS = ['skip', 'full', 'linear', 'circular']
ENTANGLEMENT_GATES = ['cx', 'cz', 'rzx']
ROTATION_GATES = ['rx', 'ry', 'rz']

class QuantumEncoderDecoder:
    hidden_state_weight_param = Parameter('Hidden State Weight')

    def __init__(self, num_qubits, config, is_recurrent=False):
        self.num_qubits = num_qubits
        self.num_blocks = config.get('num_blocks', 1)
        self.entanglement_topology = config.get('entanglement_topology', 'full')
        self.entanglement_gate = config.get('entanglement_gate', 'cx')
        self.bottleneck_size = config.get('bottleneck_size', num_qubits//2)
        self.is_recurrent = is_recurrent
        self.enforce_bottleneck = config.get('enforce_bottleneck', True)
        self.feature_encoding = config.get('feature_encoding', 'arctan_ry')
        self.quantum_rotation_family = config.get('quantum_rotation_family', 'xyz')
        if self.quantum_rotation_family not in ('xyz', 'ry'):
            raise ValueError('quantum_rotation_family must be xyz or ry')
        if not 1 <= self.bottleneck_size <= num_qubits:
            raise ValueError('bottleneck_size must be between 1 and num_qubits')
        if self.feature_encoding not in ('arctan_ry', 'bounded_ry'):
            raise ValueError('feature_encoding must be arctan_ry or bounded_ry')

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
            if self.hidden_state is not None:
                weight = 1.0 / (1.0 + np.exp(-self.hidden_weight))
                bottleneck_dm = DensityMatrix(
                    (1-weight)*bottleneck_dm.data + weight*self.hidden_state.data
                )
        # Reset is a trace-preserving discard-and-replace channel, not postselection.
        decoder_input = self.compress_bottleneck(bottleneck_dm)
        if self.is_recurrent:
            self.hidden_state = decoder_input
        predicted_state = decoder_input.evolve(self.decoder_bound)
        return bottleneck_dm, predicted_state

    def compress_bottleneck(self, bottleneck_dm):
        trash_indices = self.get_trash_indices(bottleneck_dm)
        if self.enforce_bottleneck and trash_indices:
            return bottleneck_dm.reset(trash_indices)
        return bottleneck_dm

    def prepare_state(self, state):
        features = np.asarray(state, dtype=float)
        if features.shape != (self.num_qubits,) or not np.all(np.isfinite(features)):
            raise ValueError('Expected one finite feature per qubit')
        if self.feature_encoding == 'bounded_ry':
            if np.any(np.abs(features) > 1 + 1e-12):
                raise ValueError('bounded_ry requires features in [-1, 1]')
            angles = np.arccos(np.clip(features, -1, 1))
        else:
            angles = np.pi/2 + np.arctan(features)
        param_values = {parameter: angles[index] for index, parameter in enumerate(self.input_params)}
        return Statevector.from_instruction(self.embedder.assign_parameters(param_values))

    def collapse_state(self, state):
        if isinstance(state, DensityMatrix):
            dm = state
        else:
            dm = DensityMatrix(state)

        z_expectations = []
        for qubit in range(self.num_qubits):
            pauli_list = ['I'] * self.num_qubits
            pauli_list[self.num_qubits - 1 - qubit] = 'Z'

            ex = np.real(dm.expectation_value(Pauli(''.join(pauli_list))))
            ex = max(-1.0, min(1.0, ex))
            z_expectations.append(ex)
        if self.feature_encoding == 'bounded_ry':
            return np.array(z_expectations)
        angles = np.arccos(np.clip(z_expectations, -1 + 1e-12, 1 - 1e-12))
        return np.tan(angles - np.pi/2)

    def latent_features(self, bottleneck_dm):
        """Retained single-qubit Z readouts; these are not the full quantum state."""
        retained_readouts = []
        for qubit in range(self.bottleneck_size):
            pauli_label = ['I'] * self.num_qubits
            pauli_label[self.num_qubits - 1 - qubit] = 'Z'
            retained_readouts.append(float(np.real(bottleneck_dm.expectation_value(Pauli(''.join(pauli_label))))))
        return np.asarray(retained_readouts)

    def get_trash_indices(self, bottleneck_dm):
        return list(range(self.bottleneck_size, self.num_qubits))

    def create_embedding_circuit(self):
        """
        Apply rotation gate to each qubit for embedding of classical data.
        """
        self.input_params = []
        self.embedder = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            parameter = Parameter('Embedding RY ' + str(i))
            self.input_params.append(parameter)
            self.embedder.ry(parameter, i)

    def add_entanglement_topology(self, qc: QuantumCircuit):
        if self.entanglement_topology == 'none':
            return
        elif self.entanglement_topology == 'skip':
            i = 0
            while i + 1 < self.num_qubits:
                if self.entanglement_gate.lower() == 'cx':
                    qc.cx(i, i+1)
                elif self.entanglement_gate.lower() == 'cz':
                    qc.cz(i, i+1)
                elif self.entanglement_gate.lower() == 'rzx':
                    qc.rzx(np.pi/4, i, i+1)
                else:
                    raise Exception("Unknown entanglement gate: " + self.entanglement_gate)
                i += 2
        elif self.entanglement_topology == 'full':
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

        Each block applies tied single-qubit angles before and after its
        entangling layer. The default family uses Rx, Ry and Rz; the real
        circuit ablation uses Ry only with the same angle count.
        """
        self.encoder = QuantumCircuit(self.num_qubits)
        self.trainable_params = []
        if self.is_recurrent:
            self.trainable_params.append(self.hidden_state_weight_param)
        for layer in range(self.num_blocks):
            params = []
            for i in range(self.num_qubits):
                params.append(self.add_rotation_gates(self.encoder, 'Encoder Pre-Layer ' + str(layer) + ' Rθ ' + str(i), i))
            self.add_entanglement_topology(self.encoder)
            for i in range(self.num_qubits):
                self.add_rotation_gates(self.encoder, 'Encoder Post-Layer ' + str(layer) + ' Rθ ' + str(i), i, params[i])
        self.trainable_params.extend(self.encoder.parameters)

        # The last n-k qubits are discarded and replaced with |0> before decoding.
        self.decoder = QuantumCircuit(self.num_qubits)
        for layer in range(self.num_blocks):
            params = []
            for i in range(self.num_qubits):
                params.append(self.add_rotation_gates(self.decoder, 'Decoder Pre-Layer ' + str(layer) + ' Rθ ' + str(i), i))
            self.add_entanglement_topology(self.decoder)
            for i in range(self.num_qubits):
                self.add_rotation_gates(self.decoder, 'Decoder Post-Layer ' + str(layer) + ' Rθ ' + str(i), i, params[i])
        self.trainable_params.extend(self.decoder.parameters)
        return self.encoder.compose(self.decoder)

    def add_rotation_gates(self, circuit, description, qubit_index, param=None):
        parameter = Parameter(f'{description}')
        if param is not None:
            parameter = param
        if self.quantum_rotation_family == 'xyz':
            circuit.rx(parameter, qubit_index)
        circuit.ry(parameter, qubit_index)
        if self.quantum_rotation_family == 'xyz':
            circuit.rz(parameter, qubit_index)
        return parameter

    def set_params(self, params_dict):
        encoder_params = {k: v for k,v in params_dict.items() if k in self.encoder.parameters}
        decoder_params = {k: v for k,v in params_dict.items() if k in self.decoder.parameters}
        self.encoder_bound = self.encoder.assign_parameters(encoder_params)
        self.decoder_bound = self.decoder.assign_parameters(decoder_params)
        if self.hidden_state_weight_param in params_dict:
            self.hidden_weight = float(params_dict[self.hidden_state_weight_param])
        all_params = {k: v for k,v in params_dict.items() if k in self.full_circuit.parameters}
        self.full_circuit_bound = self.full_circuit.assign_parameters(all_params)

    def load(self, fname: str):
        """
        Load a previously-saved .qpy file containing the full_circuit,
        then split it back into embedder, encoder, and decoder subcircuits.
        """
        with open(fname + '.qpy', 'rb') as fd:
            loaded_circuits = qpy.load(fd)
        if not loaded_circuits:
            raise ValueError(f'No circuits found in {fname}.qpy')
        self.full_circuit = loaded_circuits[0]
        metadata = self.full_circuit.metadata or {}
        if metadata.get('protocol_version') != 2:
            raise ValueError('Historical quantum checkpoints require the original code; retrain under protocol 2')
        expected_config = self.checkpoint_config()
        if metadata.get('model_config') != expected_config:
            raise ValueError('Checkpoint model configuration does not match this model')

        # instruction counts
        len_embed = len(self.embedder.data)
        len_encoder = len(self.encoder.data)
        len_decoder = len(self.decoder.data)

        all_ops = self.full_circuit.data
        self.embedder.data = all_ops[:len_embed]
        self.encoder.data = all_ops[len_embed : len_embed + len_encoder]
        self.decoder.data = all_ops[len_embed + len_encoder : len_embed + len_encoder + len_decoder]

        self.encoder_bound = self.encoder
        self.decoder_bound = self.decoder

        self.input_params = sorted(
            list(self.embedder.parameters),
            key=lambda p: int(p.name.split()[-1])
        )

        if self.is_recurrent:
            self.hidden_weight = metadata['hidden_weight']
        self.reset_hidden_state()

    def checkpoint_config(self):
        configuration = dict(num_qubits=self.num_qubits, num_blocks=self.num_blocks,
                    bottleneck_size=self.bottleneck_size, is_recurrent=self.is_recurrent,
                    entanglement_topology=self.entanglement_topology,
                    entanglement_gate=self.entanglement_gate,
                    feature_encoding=self.feature_encoding,
                    enforce_bottleneck=self.enforce_bottleneck)
        # Preserve the metadata schema of existing default-family checkpoints.
        if self.quantum_rotation_family != 'xyz':
            configuration['quantum_rotation_family'] = self.quantum_rotation_family
        return configuration

    def save(self, fname):
        self.full_circuit_bound.metadata = {
            'protocol_version': 2, 'model_config': self.checkpoint_config(),
            'hidden_weight': self.hidden_weight,
        }
        with open(fname + '.qpy', 'wb') as file:
            qpy.dump(self.full_circuit_bound, file)


class RingGivensRotationLayer(nn.Module):
    """
    An SO(n)-group layer built from n Givens rotation angles reused in a ring, so
    that each of the n features participates in exactly two rotations: one with
    its successor and one with its predecessor (wrapping around).

    This uses exactly n parameters, preserves orthogonality (R^T R = I, det=+1),
    and couples all features through sequential plane rotations.
    """
    def __init__(self, num_params: int):
        super().__init__()
        self.num_params = num_params
        self.angles = nn.Parameter(torch.randn(self.num_params))
        # Precompute the sequence of (i,j) planes in a ring
        self.planes = [(i, (i+1) % self.num_params) for i in range(self.num_params)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotations = torch.eye(self.num_params, device=x.device, dtype=x.dtype)
        # sequentially apply each Givens rotation
        for (i, j), angle in zip(self.planes, self.angles):
            c = torch.cos(angle)
            s = torch.sin(angle)
            givens_rotation = torch.eye(self.num_params, device=x.device, dtype=x.dtype)
            # 2×2 block in the (i,j) plane
            givens_rotation[i, i] = c;  givens_rotation[j, i] = s
            givens_rotation[i, j] = -s; givens_rotation[j, j] = c
            rotations = rotations @ givens_rotation
        return x @ rotations.T


class ClassicalEncoderDecoder(nn.Module):
    def __init__(self, num_features, config, is_recurrent=False):
        super(ClassicalEncoderDecoder, self).__init__()
        self.num_features = num_features
        self.is_recurrent = is_recurrent
        self.num_blocks = config.get('num_blocks', 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.encoder.append(RingGivensRotationLayer(num_features))
            # Decoder receives only fixed retained coordinates.
            self.decoder.append(RingGivensRotationLayer(num_features))
        self.bottleneck_size = config.get('bottleneck_size', self.num_features//2)
        self.enforce_bottleneck = config.get('enforce_bottleneck', True)
        if not 1 <= self.bottleneck_size <= num_features:
            raise ValueError('bottleneck_size must be between 1 and num_features')
        if self.is_recurrent:
            # always start at zero to give better starting gradient
            self.hidden_weight = nn.Parameter(torch.tensor([0.0]))
        self.hidden_state = None

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
        bottleneck_state, output = self.forward_tensor(x)
        return bottleneck_state.detach().cpu().numpy(), output.detach().cpu().numpy()

    def forward_tensor(self, x):
        """Differentiable forward pass with no hidden-state path around compression."""
        bottleneck_state = x
        for block in self.encoder:
            bottleneck_state = block(bottleneck_state)

        if self.is_recurrent:
            if self.hidden_state is not None:
                weight = torch.sigmoid(self.hidden_weight)
                bottleneck_state = (1-weight)*bottleneck_state + weight*self.hidden_state
        output = bottleneck_state
        if self.enforce_bottleneck:
            output = torch.cat((bottleneck_state[:self.bottleneck_size],
                                torch.zeros_like(bottleneck_state[self.bottleneck_size:])))
        if self.is_recurrent:
            self.hidden_state = output
        for block in self.decoder:
            output = block(output)
        return bottleneck_state, output

    def latent_features(self, bottleneck_state):
        return np.asarray(bottleneck_state)[:self.bottleneck_size]

    def prepare_state(self, state):
        parameter = next(self.parameters())
        return torch.as_tensor(state, dtype=parameter.dtype, device=parameter.device)

    def get_trash_indices(self, bottleneck_state):
        return list(range(self.bottleneck_size, self.num_features))

    def set_params(self, params_dict):
        for p, v in params_dict.items():
            with torch.no_grad():
                p.copy_(torch.as_tensor(v, dtype=p.dtype, device=p.device))

    def checkpoint_config(self):
        return dict(num_features=self.num_features, num_blocks=self.num_blocks,
                    bottleneck_size=self.bottleneck_size,
                    enforce_bottleneck=self.enforce_bottleneck, is_recurrent=self.is_recurrent)

    def load(self, fname):
        checkpoint = torch.load(fname + '.pth', weights_only=True)
        if checkpoint.get('protocol_version') != 2:
            raise ValueError('Historical checkpoints require the historical implementation')
        if checkpoint['model_config'] != self.checkpoint_config():
            raise ValueError('Checkpoint configuration does not match model')
        self.load_state_dict(checkpoint['state_dict'])
        self.reset_hidden_state()

    def save(self, fname):
        torch.save(dict(protocol_version=2, model_config=self.checkpoint_config(),
                        state_dict=self.state_dict()), fname + '.pth')


BOUNDED_RY_ROTATION_FAMILIES = ('xyz', 'ry')


def bounded_ry_bloch_vectors(features: torch.Tensor) -> torch.Tensor:
    """Bloch vectors (x, y, z) of RY(arccos f)|0> for bounded features f in [-1, 1].

    This is the single-qubit state QuantumEncoderDecoder prepares with
    feature_encoding='bounded_ry'. Its z component (the Z expectation) equals the
    feature itself, and its x component is sqrt(1 - f**2).
    """
    clipped_features = torch.clamp(features, -1.0, 1.0)
    transverse_components = torch.sqrt(torch.clamp(1.0 - clipped_features**2, min=0.0))
    return torch.stack((transverse_components, torch.zeros_like(clipped_features), clipped_features), dim=-1)


def bloch_rotation_matrices(angles: torch.Tensor, rotation_family: str) -> torch.Tensor:
    """SO(3) action on Bloch vectors of the tied single-qubit gates in create_qed_circuit.

    The circuit applies Rx(angle), then Ry(angle), then Rz(angle) to a qubit for the
    'xyz' family, or Ry(angle) alone for the 'ry' family. Each gate exp(-i angle P / 2)
    rotates the Bloch vector by the angle about axis P, so the composite rotation is
    R_z R_y R_x. Returns one 3 x 3 matrix per angle.
    """
    cosines, sines = torch.cos(angles), torch.sin(angles)
    zeros, ones = torch.zeros_like(angles), torch.ones_like(angles)
    rotation_about_y = torch.stack((torch.stack((cosines, zeros, sines), dim=-1),
                                    torch.stack((zeros, ones, zeros), dim=-1),
                                    torch.stack((-sines, zeros, cosines), dim=-1)), dim=-2)
    if rotation_family == 'ry':
        return rotation_about_y
    rotation_about_x = torch.stack((torch.stack((ones, zeros, zeros), dim=-1),
                                    torch.stack((zeros, cosines, -sines), dim=-1),
                                    torch.stack((zeros, sines, cosines), dim=-1)), dim=-2)
    rotation_about_z = torch.stack((torch.stack((cosines, -sines, zeros), dim=-1),
                                    torch.stack((sines, cosines, zeros), dim=-1),
                                    torch.stack((zeros, zeros, ones), dim=-1)), dim=-2)
    return rotation_about_z @ rotation_about_y @ rotation_about_x


def entanglement_topology_edges(num_units, entanglement_topology):
    """Unit pairs in the order QuantumEncoderDecoder.add_entanglement_topology places its gates."""
    if entanglement_topology == 'none':
        edges = []
    elif entanglement_topology == 'skip':
        edges = [(unit, unit + 1) for unit in range(0, num_units - 1, 2)]
    elif entanglement_topology == 'full':
        edges = [(first, second) for first in range(num_units) for second in range(first + 1, num_units)]
    elif entanglement_topology == 'linear':
        edges = [(unit, unit + 1) for unit in range(num_units - 1)]
    elif entanglement_topology == 'circular':
        edges = [(unit, unit + 1) for unit in range(num_units - 1)] + [(num_units - 1, 0)]
    else:
        raise ValueError('Unknown entanglement topology: ' + str(entanglement_topology))
    return [(first, second) for first, second in edges if first != second]


class NamedAngleHandle:
    """Hashable handle for one scalar of a parameter tensor.

    Handles carry the same names as the quantum circuit parameters, so code written
    for QuantumEncoderDecoder parameter handles, such as feature-neutral decoder
    calibration, can address the matching angles of the classical counterpart.
    """
    def __init__(self, parameter_tensor, flat_index, name):
        self.parameter_tensor = parameter_tensor
        self.flat_index = flat_index
        self.name = name

    def assign(self, value):
        with torch.no_grad():
            self.parameter_tensor.view(-1)[self.flat_index] = float(value)

    def detach(self):
        return self.parameter_tensor.detach().view(-1)[self.flat_index]

    def __repr__(self):
        return f'NamedAngleHandle({self.name!r})'


class BlochLiftedGivensEncoderDecoder(nn.Module):
    """Classical Givens counterpart of QuantumEncoderDecoder with bounded RY encoding.

    Each feature is lifted to the Bloch vector of its bounded RY encoding (see
    bounded_ry_bloch_vectors), so this model receives the same single-qubit
    representation as the quantum model. Every block mirrors create_qed_circuit: tied
    per-unit rotations, a coupling ring where the entangling gates sit and the same
    tied rotations again. On one qubit's Bloch vector the tied Rx, Ry and Rz gates act
    as exact SO(3) rotations, so those steps are identical in both models.

    The coupling is the only replaced operation: fixed Givens rotations by
    coupling_angle between the transverse (x and y) components of neighbouring units,
    over the same unit pairs and in the same order as the entangling gates. Like CZ, it
    has no trainable parameters and leaves every z component unchanged. Discarded units
    are reset to (0, 0, 1), the Bloch vector of |0>, and predictions are the z
    components, matching the quantum reset and Z readouts.

    With entanglement_topology='none' this model reproduces the non-entangling quantum
    model exactly, because product states evolve as independent single-qubit rotations.
    Each retained unit carries three real numbers: the single-qubit part of the
    retained quantum state without its multi-qubit correlations. Mixing transverse
    components does not keep each unit at unit Bloch norm, so readouts are not
    confined to [-1, 1] as quantum expectations are.
    """
    feature_encoding = 'bounded_ry'

    def __init__(self, num_features, config, is_recurrent=False):
        super().__init__()
        self.num_features = num_features
        self.is_recurrent = is_recurrent
        self.num_blocks = config.get('num_blocks', 1)
        self.bottleneck_size = config.get('bottleneck_size', num_features//2)
        self.enforce_bottleneck = config.get('enforce_bottleneck', True)
        self.rotation_family = config.get('quantum_rotation_family', 'xyz')
        self.entanglement_topology = config.get('entanglement_topology', 'full')
        self.coupling_angle = float(config.get('classical_coupling_angle', np.pi/4))
        if self.rotation_family not in BOUNDED_RY_ROTATION_FAMILIES:
            raise ValueError('quantum_rotation_family must be xyz or ry')
        if not 1 <= self.bottleneck_size <= num_features:
            raise ValueError('bottleneck_size must be between 1 and num_features')
        if not np.isfinite(self.coupling_angle):
            raise ValueError('classical_coupling_angle must be finite')
        if self.is_recurrent:
            # Registered first so the hidden-state weight leads the trainable
            # parameters, as it does in QuantumEncoderDecoder.
            self.hidden_weight = nn.Parameter(torch.tensor([0.0]))
        self.encoder_angles = nn.Parameter(torch.randn(self.num_blocks, num_features))
        self.decoder_angles = nn.Parameter(torch.randn(self.num_blocks, num_features))
        self.coupling_edges = entanglement_topology_edges(num_features, self.entanglement_topology)
        self.register_buffer('transverse_coupling_matrix', self.build_transverse_coupling_matrix(),
                             persistent=False)
        self.hidden_state = None

    @property
    def num_qubits(self):
        """Unit count under the quantum model's attribute name, for shared calibration code."""
        return self.num_features

    @property
    def trainable_params(self):
        """Named scalar handles in the quantum model's order: hidden weight, encoder angles, decoder angles."""
        handles = []
        if self.is_recurrent:
            handles.append(NamedAngleHandle(self.hidden_weight, 0, 'Hidden State Weight'))
        for angles, prefix in ((self.encoder_angles, 'Encoder'), (self.decoder_angles, 'Decoder')):
            for block_index in range(self.num_blocks):
                for unit_index in range(self.num_features):
                    handles.append(NamedAngleHandle(angles, block_index*self.num_features + unit_index,
                                                    f'{prefix} Pre-Layer {block_index} Rθ {unit_index}'))
        return handles

    def build_transverse_coupling_matrix(self):
        coupling_matrix = torch.eye(self.num_features, dtype=torch.get_default_dtype())
        cosine, sine = float(np.cos(self.coupling_angle)), float(np.sin(self.coupling_angle))
        for first_unit, second_unit in self.coupling_edges:
            givens_rotation = torch.eye(self.num_features, dtype=torch.get_default_dtype())
            givens_rotation[first_unit, first_unit] = cosine
            givens_rotation[first_unit, second_unit] = -sine
            givens_rotation[second_unit, first_unit] = sine
            givens_rotation[second_unit, second_unit] = cosine
            coupling_matrix = givens_rotation @ coupling_matrix
        return coupling_matrix

    def reset_hidden_state(self):
        self.hidden_state = None

    def set_params(self, params_dict):
        for handle, value in params_dict.items():
            handle.assign(value)

    def prepare_state(self, state):
        parameter = next(self.parameters())
        features = torch.as_tensor(np.asarray(state, dtype=float), dtype=parameter.dtype, device=parameter.device)
        if features.shape[-1] != self.num_features or not bool(torch.all(torch.isfinite(features))):
            raise ValueError('Expected one finite feature per unit')
        if bool(torch.any(torch.abs(features) > 1 + 1e-12)):
            raise ValueError('The bounded RY lift requires features in [-1, 1]')
        return bounded_ry_bloch_vectors(features)

    def rotate_units(self, unit_bloch_vectors, block_angles):
        rotation_matrices = bloch_rotation_matrices(block_angles, self.rotation_family)
        return torch.einsum('uab,...ub->...ua', rotation_matrices, unit_bloch_vectors)

    def couple_units(self, unit_bloch_vectors):
        coupled_transverse_components = torch.einsum('uv,...vc->...uc', self.transverse_coupling_matrix,
                                                     unit_bloch_vectors[..., :2])
        return torch.cat((coupled_transverse_components, unit_bloch_vectors[..., 2:]), dim=-1)

    def apply_blocks(self, unit_bloch_vectors, angles):
        for block_index in range(self.num_blocks):
            unit_bloch_vectors = self.rotate_units(unit_bloch_vectors, angles[block_index])
            unit_bloch_vectors = self.couple_units(unit_bloch_vectors)
            unit_bloch_vectors = self.rotate_units(unit_bloch_vectors, angles[block_index])
        return unit_bloch_vectors

    def compress_bottleneck(self, unit_bloch_vectors):
        if not self.enforce_bottleneck or self.bottleneck_size == self.num_features:
            return unit_bloch_vectors
        reset_units = torch.zeros_like(unit_bloch_vectors[..., self.bottleneck_size:, :])
        reset_units[..., 2] = 1.0
        return torch.cat((unit_bloch_vectors[..., :self.bottleneck_size, :], reset_units), dim=-2)

    def forward_tensor(self, unit_bloch_vectors):
        """Differentiable pass; returns the encoder output and the z readouts of the decoder output."""
        bottleneck_state = self.apply_blocks(unit_bloch_vectors, self.encoder_angles)
        if self.is_recurrent and self.hidden_state is not None:
            hidden_weight = torch.sigmoid(self.hidden_weight)
            bottleneck_state = (1 - hidden_weight)*bottleneck_state + hidden_weight*self.hidden_state
        decoder_input = self.compress_bottleneck(bottleneck_state)
        if self.is_recurrent:
            self.hidden_state = decoder_input
        decoded_state = self.apply_blocks(decoder_input, self.decoder_angles)
        return bottleneck_state, decoded_state[..., 2]

    def forward(self, unit_bloch_vectors):
        bottleneck_state, predictions = self.forward_tensor(unit_bloch_vectors)
        return bottleneck_state.detach().cpu().numpy(), predictions.detach().cpu().numpy()

    def collapse_state(self, predictions):
        """Predictions are already z readouts; kept for interface parity with the quantum model."""
        return np.asarray(predictions)

    def latent_features(self, bottleneck_state):
        """Z components of the retained units, matching the quantum model's retained Z readouts."""
        return np.asarray(bottleneck_state)[..., :self.bottleneck_size, 2]

    def get_trash_indices(self, bottleneck_state):
        return list(range(self.bottleneck_size, self.num_features))

    def checkpoint_config(self):
        return dict(num_features=self.num_features, num_blocks=self.num_blocks,
                    bottleneck_size=self.bottleneck_size, enforce_bottleneck=self.enforce_bottleneck,
                    is_recurrent=self.is_recurrent, feature_encoding='bounded_ry_bloch_lift',
                    rotation_family=self.rotation_family, entanglement_topology=self.entanglement_topology,
                    coupling_angle=self.coupling_angle)

    def load(self, fname):
        checkpoint = torch.load(fname + '.pth', weights_only=True)
        if checkpoint.get('protocol_version') != 2:
            raise ValueError('Historical checkpoints require the historical implementation')
        if checkpoint['model_config'] != self.checkpoint_config():
            raise ValueError('Checkpoint configuration does not match model')
        self.load_state_dict(checkpoint['state_dict'])
        self.reset_hidden_state()

    def save(self, fname):
        torch.save(dict(protocol_version=2, model_config=self.checkpoint_config(),
                        state_dict=self.state_dict()), fname + '.pth')
