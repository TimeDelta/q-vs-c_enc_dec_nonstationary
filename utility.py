from functools import reduce
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix

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