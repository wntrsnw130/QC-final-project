import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from qfs.qubo_constructor import construct_qubo_matrix
from qfs.alpha_binary_search_with_sa import alpha_binary_search_with_sa
from qfs.solver import simulated_annealing
from qfs.mi_estimation import compute_importance_classification, compute_importance_regression, compute_redundancy
from data.synth_generator import generate_synth10_classification, generate_synth10_regression
from data.real_data_loader import load_classification_dataset, load_regression_dataset

def benchmark_classification(n_runs=10, k=4):
    acc_random, acc_all, acc_mi, acc_qfs = [], [], [], []
    for _ in range(n_runs):
        X, y, _ = generate_synth10_classification()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        n = X.shape[1]

        # Random
        random_idx = np.random.choice(X_train.shape[1], k, replace=False)
        clf = RandomForestClassifier()
        clf.fit(X_train[:, random_idx], y_train)
        y_pred = clf.predict(X_test[:, random_idx])
        acc_random.append(accuracy_score(y_test, y_pred))

        # All features
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_all.append(accuracy_score(y_test, y_pred))

        # MI
        I = compute_importance_classification(X_train, y_train)
        mi_idx = np.argsort(I)[::-1][:k]
        clf.fit(X_train[:, mi_idx], y_train)
        y_pred = clf.predict(X_test[:, mi_idx])
        acc_mi.append(accuracy_score(y_test, y_pred))

        # QFS
        R = compute_redundancy(X_train)
        alpha, x_mask = alpha_binary_search_with_sa(I, R, k)
        qfs_idx = np.where(x_mask == 1)[0]
        clf.fit(X_train[:, qfs_idx], y_train)
        y_pred = clf.predict(X_test[:, qfs_idx])
        acc_qfs.append(accuracy_score(y_test, y_pred))

    print(f"Random Features Accuracy: {np.mean(acc_random):.4f} ± {np.std(acc_random):.4f}")
    print(f"All Features Accuracy: {np.mean(acc_all):.4f}")
    print(f"MI Top-{k} Accuracy: {np.mean(acc_mi):.4f}")
    print(f"QFS Selected Accuracy: {np.mean(acc_qfs):.4f}")

def benchmark_regression(n_runs=10, k=4):
    mse_random, mse_all, mse_mi, mse_qfs = [], [], [], []
    for _ in range(n_runs):
        X, y, _ = generate_synth10_regression()
        n = X.shape[1]

        # Random
        random_idx = np.random.choice(n, k, replace=False)
        reg = RandomForestRegressor()
        reg.fit(X[:, random_idx], y)
        y_pred = reg.predict(X[:, random_idx])
        mse_random.append(mean_squared_error(y, y_pred))

        # All features
        reg.fit(X, y)
        y_pred = reg.predict(X)
        mse_all.append(mean_squared_error(y, y_pred))

        # MI
        I = compute_importance_regression(X, y)
        mi_idx = np.argsort(I)[::-1][:k]
        reg.fit(X[:, mi_idx], y)
        y_pred = reg.predict(X[:, mi_idx])
        mse_mi.append(mean_squared_error(y, y_pred))

        # QFS
        R = compute_redundancy(X)
        alpha, x_mask = alpha_binary_search_with_sa(I, R, k)
        qfs_idx = np.where(x_mask == 1)[0]
        reg.fit(X[:, qfs_idx], y)
        y_pred = reg.predict(X[:, qfs_idx])
        mse_qfs.append(mean_squared_error(y, y_pred))

    print(f"Random Features MSE: {np.mean(mse_random):.2f} ± {np.std(mse_random):.2f}")
    print(f"All Features MSE: {np.mean(mse_all):.2f}")
    print(f"MI Top-{k} MSE: {np.mean(mse_mi):.2f}")
    print(f"QFS Selected MSE: {np.mean(mse_qfs):.2f}")

def benchmark_classification_real(dataset_name, k=4, n_runs=5):
    acc_random, acc_all, acc_mi, acc_qfs = [], [], [], []
    X, y = load_classification_dataset(dataset_name)
    n = X.shape[1]

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        # Random
        random_idx = np.random.choice(n, k, replace=False)
        clf = RandomForestClassifier()
        clf.fit(X_train[:, random_idx], y_train)
        y_pred = clf.predict(X_test[:, random_idx])
        acc_random.append(accuracy_score(y_test, y_pred))

        # All features
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_all.append(accuracy_score(y_test, y_pred))

        # MI
        I = compute_importance_classification(X_train, y_train)
        mi_idx = np.argsort(I)[::-1][:k]
        clf.fit(X_train[:, mi_idx], y_train)
        y_pred = clf.predict(X_test[:, mi_idx])
        acc_mi.append(accuracy_score(y_test, y_pred))

        # QFS
        R = compute_redundancy(X_train)
        alpha, x_mask = alpha_binary_search_with_sa(I, R, k)
        qfs_idx = np.where(x_mask == 1)[0]
        clf.fit(X_train[:, qfs_idx], y_train)
        y_pred = clf.predict(X_test[:, qfs_idx])
        acc_qfs.append(accuracy_score(y_test, y_pred))

    print(f"[{dataset_name}] Random Accuracy: {np.mean(acc_random):.4f} ± {np.std(acc_random):.4f}")
    print(f"[{dataset_name}] All Features Accuracy: {np.mean(acc_all):.4f}")
    print(f"[{dataset_name}] MI Top-{k} Accuracy: {np.mean(acc_mi):.4f}")
    print(f"[{dataset_name}] QFS Accuracy: {np.mean(acc_qfs):.4f}")

def benchmark_regression_real(dataset_name, k=4, n_runs=5):
    mse_random, mse_all, mse_mi, mse_qfs = [], [], [], []
    X, y = load_regression_dataset(dataset_name)
    n = X.shape[1]

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        # Random
        random_idx = np.random.choice(n, k, replace=False)
        reg = RandomForestRegressor()
        reg.fit(X_train[:, random_idx], y_train)
        y_pred = reg.predict(X_test[:, random_idx])
        mse_random.append(mean_squared_error(y_test, y_pred))

        # All features
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse_all.append(mean_squared_error(y_test, y_pred))

        # MI
        I = compute_importance_regression(X_train, y_train)
        mi_idx = np.argsort(I)[::-1][:k]
        reg.fit(X_train[:, mi_idx], y_train)
        y_pred = reg.predict(X_test[:, mi_idx])
        mse_mi.append(mean_squared_error(y_test, y_pred))

        # QFS
        R = compute_redundancy(X_train)
        alpha, x_mask = alpha_binary_search_with_sa(I, R, k)
        qfs_idx = np.where(x_mask == 1)[0]
        reg.fit(X_train[:, qfs_idx], y_train)
        y_pred = reg.predict(X_test[:, qfs_idx])
        mse_qfs.append(mean_squared_error(y_test, y_pred))

    print(f"[{dataset_name}] Random MSE: {np.mean(mse_random):.2f} ± {np.std(mse_random):.2f}")
    print(f"[{dataset_name}] All Features MSE: {np.mean(mse_all):.2f}")
    print(f"[{dataset_name}] MI Top-{k} MSE: {np.mean(mse_mi):.2f}")
    print(f"[{dataset_name}] QFS MSE: {np.mean(mse_qfs):.2f}")


if __name__ == '__main__':
    print("===== Classification Benchmark =====")
    benchmark_classification()
    print("===== Regression Benchmark =====")
    benchmark_regression()
