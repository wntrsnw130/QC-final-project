import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from qfs.mi_estimation import compute_importance_classification, compute_importance_regression, compute_redundancy
from qfs.alpha_binary_search_with_sa import alpha_binary_search_with_sa

from data.synth_generator import generate_synth10_classification, generate_synth10_regression
from data.real_data_loader import load_classification_dataset, load_regression_dataset

from tqdm import tqdm


def sensitivity_analysis_classification_generic(X, y, dataset_name, k_list, n_runs=3):
    n = X.shape[1]
    valid_k_list = []
    acc_random_list, acc_mi_list, acc_qfs_list = [], [], []

    print(f"\n>>> Running classification sensitivity on {dataset_name} ({n} features)")
    start_time = time.time()

    for k in tqdm(k_list, desc=f"[{dataset_name}] Processing k"):
        if k > n:
            print(f"Warning: k={k} is larger than feature count n={n}, skip.")
            continue

        valid_k_list.append(k)
        acc_random, acc_mi, acc_qfs = [], [], []

        for _ in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Random
            random_idx = np.random.choice(n, k, replace=False)
            clf = RandomForestClassifier()
            clf.fit(X_train[:, random_idx], y_train)
            acc_random.append(accuracy_score(y_test, clf.predict(X_test[:, random_idx])))

            # MI
            I = compute_importance_classification(X_train, y_train)
            mi_idx = np.argsort(I)[::-1][:k]
            clf.fit(X_train[:, mi_idx], y_train)
            acc_mi.append(accuracy_score(y_test, clf.predict(X_test[:, mi_idx])))

            # QFS
            R = compute_redundancy(X_train)
            n_iter_sa = 100 if n > 20 else 300
            alpha, x_mask = alpha_binary_search_with_sa(I, R, k, n_iter_sa=n_iter_sa)
            qfs_idx = np.where(x_mask == 1)[0]
            clf.fit(X_train[:, qfs_idx], y_train)
            acc_qfs.append(accuracy_score(y_test, clf.predict(X_test[:, qfs_idx])))

        acc_random_list.append(np.mean(acc_random))
        acc_mi_list.append(np.mean(acc_mi))
        acc_qfs_list.append(np.mean(acc_qfs))

    _plot_sensitivity(valid_k_list, acc_random_list, acc_mi_list, acc_qfs_list, 'Accuracy', dataset_name)

    elapsed = time.time() - start_time
    print(f"\n>>> {dataset_name} finished in {elapsed:.1f} seconds")
    gc.collect()


def sensitivity_analysis_regression_generic(X, y, dataset_name, k_list, n_runs=3):
    n = X.shape[1]
    valid_k_list = []
    mse_random_list, mse_mi_list, mse_qfs_list = [], [], []

    print(f"\n>>> Running regression sensitivity on {dataset_name} ({n} features)")
    start_time = time.time()

    for k in tqdm(k_list, desc=f"[{dataset_name}] Processing k"):
        if k > n:
            print(f"Warning: k={k} is larger than feature count n={n}, skip.")
            continue

        valid_k_list.append(k)
        mse_random, mse_mi, mse_qfs = [], [], []

        for _ in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Random
            random_idx = np.random.choice(n, k, replace=False)
            reg = RandomForestRegressor()
            reg.fit(X_train[:, random_idx], y_train)
            mse_random.append(mean_squared_error(y_test, reg.predict(X_test[:, random_idx])))

            # MI
            I = compute_importance_regression(X_train, y_train)
            mi_idx = np.argsort(I)[::-1][:k]
            reg.fit(X_train[:, mi_idx], y_train)
            mse_mi.append(mean_squared_error(y_test, reg.predict(X_test[:, mi_idx])))

            # QFS
            R = compute_redundancy(X_train)
            n_iter_sa = 100 if n > 20 else 300
            alpha, x_mask = alpha_binary_search_with_sa(I, R, k, n_iter_sa=n_iter_sa)
            qfs_idx = np.where(x_mask == 1)[0]
            reg.fit(X_train[:, qfs_idx], y_train)
            mse_qfs.append(mean_squared_error(y_test, reg.predict(X_test[:, qfs_idx])))

        mse_random_list.append(np.mean(mse_random))
        mse_mi_list.append(np.mean(mse_mi))
        mse_qfs_list.append(np.mean(mse_qfs))

    _plot_sensitivity(valid_k_list, mse_random_list, mse_mi_list, mse_qfs_list, 'MSE', dataset_name, regression=True)

    elapsed = time.time() - start_time
    print(f"\n>>> {dataset_name} finished in {elapsed:.1f} seconds")
    gc.collect()


def _plot_sensitivity(k_list, random_list, mi_list, qfs_list, metric_name, dataset_name, regression=False):
    plt.figure(figsize=(8, 6))
    plt.plot(k_list, random_list, marker='o', label='Random')
    plt.plot(k_list, mi_list, marker='s', label='MI Top-k')
    plt.plot(k_list, qfs_list, marker='^', label='QFS')
    plt.xlabel('Number of Selected Features (k)')
    plt.ylabel(metric_name)
    plt.title(f'Sensitivity Analysis on {dataset_name}')
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    suffix = '_regression' if regression else ''
    plt.savefig(f'results/sensitivity_{dataset_name}{suffix}.png')
    plt.close()


def run_all_sensitivity(k_list=[3, 5, 7, 10, 15, 20], n_runs=3):
    # Classification
    datasets_classification = [('synth10', generate_synth10_classification),
                               ('ionosphere', lambda: load_classification_dataset('ionosphere')),
                               ('breast_cancer', lambda: load_classification_dataset('breast_cancer')),
                               ('digits', lambda: load_classification_dataset('digits'))]

    for name, loader in datasets_classification:
        result = loader()
        X, y = result[:2]  # 只取前兩個，無論有幾個
        sensitivity_analysis_classification_generic(X, y, name, k_list, n_runs)

    # Regression
    datasets_regression = [('synth10_regression', generate_synth10_regression),
                           ('diabetes', lambda: load_regression_dataset('diabetes')),
                           ('california', lambda: load_regression_dataset('california'))]

    for name, loader in datasets_regression:
        result = loader()
        X, y = result[:2]
        sensitivity_analysis_regression_generic(X, y, name, k_list, n_runs)


if __name__ == '__main__':
    run_all_sensitivity()
