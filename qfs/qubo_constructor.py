import numpy as np

def construct_qubo_matrix(I, R, alpha=0.5):
    n = len(I)
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                Q[i, j] = R[i, j] - alpha * (R[i, j] + I[i])
            else:
                Q[i, j] = R[i, j] - alpha * (R[i, j])
    return Q

if __name__ == '__main__':
    from data.synth_generator import generate_synth10_classification
    from qfs.mi_estimation import compute_importance_classification, compute_redundancy

    X, y, informative_idx = generate_synth10_classification()
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)

    alpha = 0.5
    Q = construct_qubo_matrix(I, R, alpha)
    print(f"QUBO Matrix (alpha={alpha}):\n{Q}")
