import numpy as np

def simulated_annealing(Q, n_iter=1000, init_temp=10.0, cooling_rate=0.995):
    n = Q.shape[0]
    x = np.random.randint(0, 2, size=n)
    best_x = x.copy()
    best_score = x @ Q @ x
    temp = init_temp

    for _ in range(n_iter):
        i = np.random.randint(n)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]  # flip bit
        score_new = x_new @ Q @ x_new
        delta = score_new - best_score
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x = x_new
            if score_new < best_score:
                best_x = x_new
                best_score = score_new
        temp *= cooling_rate

    return best_x, best_score

if __name__ == '__main__':
    from data.synth_generator import generate_synth10_classification
    from qfs.mi_estimation import compute_importance_classification, compute_redundancy
    from qfs.qubo_constructor import construct_qubo_matrix

    X, y, _ = generate_synth10_classification()
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)
    Q = construct_qubo_matrix(I, R, alpha=0.5)

    x, score = simulated_annealing(Q)
    selected_features = np.where(x == 1)[0]
    print(f"Selected features: {selected_features}")
    print(f"Final score: {score}")
