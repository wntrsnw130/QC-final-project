import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr

from qfs.mi_estimation import compute_importance_classification, compute_importance_regression, compute_redundancy
from qfs.qubo_constructor import construct_qubo_matrix
from qfs.alpha_binary_search_with_sa import alpha_binary_search_with_sa
import os

# Dataset loader map
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, fetch_california_housing, fetch_openml
from data.real_data_loader import load_ionosphere

# Dataset loader map (已移除 wine，新增 ionosphere)
dataset_loaders = {
    'breast_cancer': load_breast_cancer,
    'diabetes': load_diabetes,
    'digits': load_digits,
    'ionosphere': load_ionosphere,
    'california': fetch_california_housing
}


def analyze_dataset(name, k=4, alpha=0.5):
    print(f"\n===== Analyzing {name} =====")
    data = dataset_loaders[name]()
    X, y = data.data, data.target
    feature_names = data.feature_names if hasattr(data, 'feature_names') else [f'feat_{i}' for i in range(X.shape[1])]

    is_classification = len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer)

    if is_classification:
        I = compute_importance_classification(X, y)
    else:
        I = compute_importance_regression(X, y)

    R = compute_redundancy(X)
    _, x_mask = alpha_binary_search_with_sa(I, R, k)
    qfs_idx = np.where(x_mask == 1)[0]

    results = []
    for i in qfs_idx:
        if is_classification:
            group0 = X[y == 0][:, i]
            group1 = X[y == 1][:, i]
            t_stat, p_value = ttest_ind(group0, group1)
            pooled_std = np.sqrt((np.std(group0, ddof=1)**2 + np.std(group1, ddof=1)**2) / 2)
            cohens_d = (np.mean(group0) - np.mean(group1)) / pooled_std
            results.append({
                'Feature Index': i,
                'Feature Name': feature_names[i],
                't-stat': round(t_stat, 3),
                'p-value': round(p_value, 4),
                "Cohen's d": round(cohens_d, 3)
            })
        else:
            corr, p_value = pearsonr(X[:, i], y)
            results.append({
                'Feature Index': i,
                'Feature Name': feature_names[i],
                'Pearson r': round(corr, 3),
                'p-value': round(p_value, 4)
            })

        # plot
        df_plot = pd.DataFrame({
            'Feature Value': X[:, i],
            'Label': y
        })
        plt.figure(figsize=(6, 4))
        if is_classification:
            sns.boxplot(x='Label', y='Feature Value', data=df_plot)
        else:
            sns.scatterplot(x='Feature Value', y='Label', data=df_plot)
        plt.title(f'{name} - {feature_names[i]}')
        plt.tight_layout()
        os.makedirs(f'results/stat_plots/{name}', exist_ok=True)
        plt.savefig(f'results/stat_plots/{name}/feature_{i}_{feature_names[i]}.png')
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv(f'results/stat_plots/{name}_stat_summary.csv', index=False)
    print(df)
    return df

# Run on all datasets
if __name__ == "__main__":
    dataset_loaders = {
    'breast_cancer': load_breast_cancer,
    'diabetes': load_diabetes,
    'digits': load_digits,
    'ionosphere': load_ionosphere,
    'california': fetch_california_housing
    }

    for dataset in dataset_loaders:
        try:
            analyze_dataset(dataset)
        except Exception as e:
            print(f"[Error] Skipped {dataset} due to: {e}")
