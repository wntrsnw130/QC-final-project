from sklearn.datasets import load_iris, load_digits, load_breast_cancer, fetch_california_housing, load_diabetes
import pandas as pd

def load_ionosphere():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='ionosphere', version=1, as_frame=True)
    X = data.data.to_numpy()
    y = (data.target == 'g').astype(int)  # 將 'g', 'b' 轉成 1/0
    return X, y

def load_classification_dataset(name):
    if name == 'iris':
        data = load_iris()
    elif name == 'digits':
        data = load_digits()
    elif name == 'breast_cancer':
        data = load_breast_cancer()
    elif name == 'ionosphere':
        return load_ionosphere()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return data.data, data.target

def load_regression_dataset(name):
    if name == 'diabetes':
        data = load_diabetes()
    elif name == 'california':
        data = fetch_california_housing()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return data.data, data.target

if __name__ == '__main__':
    X, y = load_classification_dataset('ionosphere')
    print(f"Ionosphere loaded. X shape: {X.shape}, y shape: {y.shape}")
    X, y = load_regression_dataset('diabetes')
    print(f"Diabetes loaded. X shape: {X.shape}, y shape: {y.shape}")
