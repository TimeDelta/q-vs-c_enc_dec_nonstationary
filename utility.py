from functools import reduce
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix

ROUNDING_ERROR_LIMIT = 1E-8

def normalize_classical_vector(state):
    """
    Set unit L2 norm. This is the closest feasible analogue to the non-linearity introduced by
    clipping of negative density matrix eigenvalues because in order to get a spectral nonlinearity
    of a vector, we need to some how turn it into a matrix in a way that allows for negative
    eigenvalues to be clipped and reduced back into a vector without losing per-feature structure
    (a very non-trivial task).

    This approach maintains the full per-feature structure of the vector while ensuring that its
    overall scale remains fixed, making it more directly comparable to enforcing unit trace on a
    density matrix without imposing additional structure like that of a probability simplex,
    especially since the model is already using RestrictedParamCountCayleyLinear layers.
    """
    norm = torch.norm(state, p=2)
    if norm != 0:
        return state / norm
    return state

def fix_dm_array(dm_array):
    """
    WARNING: This introduces the only nonlinearity in the quantum architecture so be sure to
    enforce unit L2 norm on equivalent classical states via `normalize_classical_vector` to
    maintain a more fair comparison.
    """
    return normalize_to_unit_trace(clip_negative_eigenvalues(enforce_hermiticity(dm_array)))

def clip_negative_eigenvalues(density_matrix_array):
    """
    WARNING: This introduces the only nonlinearity in the quantum architecture so be sure to
    enforce unit L2 norm on equivalent classical states via `normalize_classical_vector` to
    maintain a more fair comparison.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(torch.tensor(density_matrix_array, dtype=torch.complex128))
    clipped_eigenvalues = torch.clamp(eigenvalues, min=ROUNDING_ERROR_LIMIT).to(dtype=torch.complex128)
    projected_dm = eigenvectors @ torch.diag(clipped_eigenvalues) @ eigenvectors.T.conj()
    return projected_dm.detach().numpy()

def enforce_hermiticity(dm_array):
    dm_array = (dm_array + dm_array.T.conj()) / 2.0
    hermiticity_error = np.linalg.norm(dm_array - dm_array.T.conj()).item()
    if hermiticity_error > ROUNDING_ERROR_LIMIT:
        print("WARNING: Hermiticity error", hermiticity_error)
    return dm_array

def normalize_to_unit_trace(state):
    trace_val = np.trace(state)
    if trace_val != 0:
        return state / trace_val
    return state

def soft_reset_trash_qubits(bottleneck_state, bottleneck_size, reset_strength=0.5):
    """
    Push the trash qubits in the bottleneck_state more towards |0>

    For each trash qubit (determined by lowest marginal p0),
    the density matrix is updated as:
        ρ_new = (1 - reset_strength) * ρ + reset_strength * (P0 ρ P0)
    where P0 = |0><0| on the appropriate qubit.
    """
    marginals = []
    for q in range(bottleneck_state.num_qubits):
        trace_indices = list(range(bottleneck_state.num_qubits))
        trace_indices.remove(q)
        dm = partial_trace(bottleneck_state, trace_indices)
        p0 = np.real(dm.data[0, 0])
        marginals.append((q, p0))

    sorted_marginals = sorted(marginals, key=lambda x: x[1])
    num_trash = bottleneck_state.num_qubits - bottleneck_size
    trash_qubit_indices = [q for (q, p0) in sorted_marginals[:num_trash]]

    dm_full = DensityMatrix(bottleneck_state)

    P0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    for q in trash_qubit_indices:
        ops = []
        for i in range(bottleneck_state.num_qubits):
            if i == (bottleneck_state.num_qubits - 1 - q):
                ops.append(P0)
            else:
                ops.append(I)
        proj = reduce(np.kron, ops)
        hard_reset_state = proj @ dm_full.data @ proj.conj().T
        new_data = (1 - reset_strength) * dm_full.data + reset_strength * hard_reset_state
        dm_full = DensityMatrix(new_data)
        dm_full = dm_full / dm_full.trace()

    return dm_full

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

def without_t_gate(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Copy without the 't' parameter
    """
    new_qc = QuantumCircuit(qc.num_qubits - 1)
    qubit_map = {old: new for old, new in zip(qc.qubits[:qc.num_qubits - 1], new_qc.qubits)}
    for instr in qc.data:
        # skip any step that depends on t
        if any(str(param) == 't' for param in instr.operation.params):
            continue
        try:
            # only seems to happen intermittently
            new_qubits = [qubit_map[q] for q in instr.qubits]
        except KeyError:
            continue # instruction involves a qubit that was dropped
        new_qc.append(instr.operation, new_qubits, instr.clbits)
    return new_qc

def has_method(obj, method_name):
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))
