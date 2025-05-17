import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qfs.qubo_constructor import construct_qubo_matrix
from qfs.mi_estimation import compute_importance_classification, compute_redundancy
from data.synth_generator import generate_synth10_classification

def plot_qubo_heatmap(Q, title='QUBO Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(Q, cmap='coolwarm', center=0, square=True, annot=False)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def demo_qubo_heatmap(alpha_list=[0.1, 0.5, 0.9]):
    # 使用 synth10 測試
    X, y, _ = generate_synth10_classification()
    I = compute_importance_classification(X, y)
    R = compute_redundancy(X)

    for alpha in alpha_list:
        Q = construct_qubo_matrix(I, R, alpha)
        plot_qubo_heatmap(Q, title=f'QUBO Matrix (alpha={alpha})')

if __name__ == '__main__':
    demo_qubo_heatmap()
