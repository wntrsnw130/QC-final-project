import numpy as np

def generate_synth10_classification(n_samples=1000, n_features=10, n_informative=4, random_state=42):
    np.random.seed(random_state)
    informative_idx = np.sort(np.random.choice(n_features, n_informative, replace=False))
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_informative)
    z = X[:, informative_idx] @ weights
    threshold = np.mean(z)
    y = (z > threshold).astype(int)
    return X, y, informative_idx

def generate_synth10_regression(n_samples=1000, n_features=10, n_informative=4, random_state=42):
    np.random.seed(random_state)
    informative_idx = np.sort(np.random.choice(n_features, n_informative, replace=False))
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_informative)
    z = X[:, informative_idx] @ weights
    noise = np.random.normal(0, 1, size=n_samples)
    y = z + noise
    return X, y, informative_idx

if __name__ == '__main__':
    X_clf, y_clf, idx_clf = generate_synth10_classification()
    X_reg, y_reg, idx_reg = generate_synth10_regression()
    print(f"Classification Informative Index: {idx_clf}")
    print(f"Regression Informative Index: {idx_reg}")
    print(f"Classification X shape: {X_clf.shape}, y shape: {y_clf.shape}")
    print(f"Regression X shape: {X_reg.shape}, y shape: {y_reg.shape}")
