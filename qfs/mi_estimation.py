import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

# use mutual_info_classif to calculate MI to the label for every feature 
# calculate the pairwise MI

def compute_importance_classification(X, y):
    return mutual_info_classif(X, y, discrete_features=False)

# use KBinsDiscretizer to dicrete label
# use MI to calculate the importance

def compute_importance_regression(X, y, n_bins=20):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()
    return mutual_info_classif(X, y_binned, discrete_features=False)

# Redundancy calculatd by mutual_info_classif

from sklearn.feature_selection import mutual_info_regression

def compute_redundancy(X):
    n_features = X.shape[1]
    R = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi = mutual_info_regression(X[:, [i]], X[:, j])
            R[i, j] = R[j, i] = mi[0]
    return R


def compute_redundancy(X):
    n_features = X.shape[1]
    R = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi = mutual_info_regression(X[:, [i]], X[:, j])
            R[i, j] = R[j, i] = mi[0]
    return R


if __name__ == '__main__':
    from data.synth_generator import generate_synth10_classification, generate_synth10_regression

    X_clf, y_clf, _ = generate_synth10_classification()
    I_clf = compute_importance_classification(X_clf, y_clf)
    R_clf = compute_redundancy(X_clf)
    print(f"[Classification] MI: {I_clf}\nRedundancy Matrix:\n{R_clf}")

    X_reg, y_reg, _ = generate_synth10_regression()
    I_reg = compute_importance_regression(X_reg, y_reg)
    R_reg = compute_redundancy(X_reg)
    print(f"[Regression] MI (binned): {I_reg}\nRedundancy Matrix:\n{R_reg}")
