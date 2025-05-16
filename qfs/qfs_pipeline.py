import numpy as np
from data.synth_generator import generate_synth10_classification
from qfs.mi_estimation import compute_importance_classification, compute_redundancy
from qfs.qubo_constructor import construct_qubo_matrix
from qfs.alpha_binary_search import alpha_binary_search
from qfs.solver import simulated_annealing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def qfs_pipeline_classification(n_samples=1000, k=4):
    # 1. 資料產生
    X, y, informative_idx = generate_synth10_classification(n_samples=n_samples)
    print(f"Ground truth informative features: {informative_idx}")

    # 2. MI 計算
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)

    # 3. alpha binary search
    alpha, x_mask = alpha_binary_search(I, R, k)
    selected_features = np.where(x_mask == 1)[0]
    print(f"Selected features by QFS: {selected_features} (Alpha: {alpha:.4f})")

    # 4. RF 分類器
    X_selected = X[:, selected_features]
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_selected, y)
    y_pred = clf.predict(X_selected)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy using selected features: {acc:.4f}")

if __name__ == '__main__':
    qfs_pipeline_classification()
