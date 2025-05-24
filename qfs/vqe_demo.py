import numpy as np
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.estimator import Estimator
from qiskit_aer.primitives import Estimator as AerEstimator

from qfs.qubo_constructor import construct_qubo_matrix
from qfs.mi_estimation import compute_importance_classification, compute_redundancy
from data.synth_generator import generate_synth10_classification


def qubo_to_ising(Q):
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        h[i] = Q[i, i] / 2
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4
    offset = np.sum(Q) / 4 + np.sum(np.diag(Q)) / 4
    return h, J, offset


def build_sparse_pauli_op(h, J):
    n = len(h)
    pauli_terms = []
    coeffs = []

    # Linear terms: Z_i
    for i in range(n):
        z = ['I'] * n
        z[i] = 'Z'
        pauli_terms.append("".join(reversed(z)))
        coeffs.append(h[i])

    # Quadratic terms: Z_i Z_j
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                z = ['I'] * n
                z[i] = 'Z'
                z[j] = 'Z'
                pauli_terms.append("".join(reversed(z)))
                coeffs.append(J[i, j])

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


def demo_vqe_qubo(alpha=0.5):
    print(f"\n✅ Running VQE demo with alpha = {alpha}")
    X, y, _ = generate_synth10_classification()
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)
    Q = construct_qubo_matrix(I, R, alpha)

    h, J, offset = qubo_to_ising(Q)
    hamiltonian = build_sparse_pauli_op(h, J)

    ansatz = TwoLocal(Q.shape[0], 'ry', 'cz', reps=2)
    optimizer = COBYLA(maxiter=100)

    backend = AerEstimator()  # or Estimator() if using basic simulator
    vqe = VQE(estimator=backend, ansatz=ansatz, optimizer=optimizer)

    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real + offset

    print(f"\n🔬 VQE estimated minimum energy: {energy:.4f}")
    print("⚡ VQE executed successfully using Qiskit modular ≥ 0.44 format")


if __name__ == '__main__':
    demo_vqe_qubo(alpha=0.5)
