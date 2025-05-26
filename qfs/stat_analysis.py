import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp

from qfs.qubo_constructor import construct_qubo_matrix
from qfs.mi_estimation import compute_importance_classification, compute_redundancy
from data.synth_generator import generate_synth10_classification


def qubo_to_ising(Q):
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    for i in range(n):
        h[i] = Q[i, i] / 2
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4
    offset = Q.sum() / 4 + np.diag(Q).sum() / 4
    return h, J, offset


def build_sparse_pauli_op(h, J):
    n = len(h)
    terms, coeffs = [], []
    for i in range(n):
        z = ["I"] * n
        z[i] = "Z"
        terms.append("".join(reversed(z)))
        coeffs.append(h[i])
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                z = ["I"] * n
                z[i] = z[j] = "Z"
                terms.append("".join(reversed(z)))
                coeffs.append(J[i, j])
    return SparsePauliOp.from_list(list(zip(terms, coeffs)))


def demo_vqe_aer(alpha=0.5):
    print(f"\n🧪 Running VQE locally with α = {alpha}")

    X, y, _ = generate_synth10_classification()
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)
    Q = construct_qubo_matrix(I, R, alpha)

    h, J, offset = qubo_to_ising(Q)
    hamiltonian = build_sparse_pauli_op(h, J)

    ansatz = TwoLocal(Q.shape[0], 'ry', 'cz', reps=1)
    optimizer = COBYLA(maxiter=100)

    backend = Aer.get_backend("aer_simulator")
    qi = QuantumInstance(backend=backend, shots=1024)

    vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real + offset

    print(f"✅ Local VQE estimated minimum energy: {energy:.4f}")


if __name__ == "__main__":
    demo_vqe_aer(alpha=0.5)
