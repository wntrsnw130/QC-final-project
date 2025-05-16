from qfs.benchmark import benchmark_classification_real, benchmark_regression_real

def study_classification():
    datasets = ['ionosphere', 'breast_cancer', 'digits']
    for dataset in datasets:
        print(f"===== Real Classification: {dataset} =====")
        benchmark_classification_real(dataset, k=4)

def study_regression():
    datasets = ['diabetes', 'california']
    for dataset in datasets:
        print(f"===== Real Regression: {dataset} =====")
        benchmark_regression_real(dataset, k=4)

if __name__ == '__main__':
    study_classification()
    study_regression()
