import numpy as np
from qfs.qubo_constructor import construct_qubo_matrix
from qfs.solver import simulated_annealing

def alpha_binary_search_with_sa(I, R, k_target, tol=1, max_iter=20, n_iter_sa=500):
    a, b = 0.0, 1.0
    for _ in range(max_iter):
        alpha = (a + b) / 2
        Q = construct_qubo_matrix(I, R, alpha)
        x, _ = simulated_annealing(Q, n_iter=n_iter_sa)
        selected = np.sum(x)
        print(f"[Alpha-SA Search] Alpha={alpha:.4f}, Selected={selected}")
        if selected > k_target + tol:
            b = alpha
        elif selected < k_target - tol:
            a = alpha
        else:
            break
    return alpha, x

if __name__ == '__main__':
    from data.synth_generator import generate_synth10_classification
    from qfs.mi_estimation import compute_importance_classification, compute_redundancy

    X, y, _ = generate_synth10_classification()
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)
    k = 4
    alpha, x = alpha_binary_search_with_sa(I, R, k)
    selected_features = np.where(x == 1)[0]
    print(f"Selected features (SA): {selected_features}")
    print(f"Alpha: {alpha}")
