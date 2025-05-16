import numpy as np
from qfs.qubo_constructor import construct_qubo_matrix

def solve_qubo_brute_force(Q):
    n = Q.shape[0]
    best_x = None
    best_score = np.inf
    for i in range(1 << n):
        x = np.array(list(np.binary_repr(i, width=n)), dtype=int)
        score = x @ Q @ x
        if score < best_score:
            best_score = score
            best_x = x
    return best_x, best_score

def alpha_binary_search(I, R, k_target, tol=1, max_iter=20):
    a, b = 0.0, 1.0
    for _ in range(max_iter):
        alpha = (a + b) / 2
        Q = construct_qubo_matrix(I, R, alpha)
        x, _ = solve_qubo_brute_force(Q)
        selected = np.sum(x)
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

    X, y, informative_idx = generate_synth10_classification()
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)
    k = 4
    alpha, x = alpha_binary_search(I, R, k)
    selected_features = np.where(x == 1)[0]
    print(f"Selected features: {selected_features}")
    print(f"Alpha: {alpha}")
