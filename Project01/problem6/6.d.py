import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import norm


def simulate_pca(cov_matrix, num_samples, variance_threshold=0.75):
    start_time = time.time()
    pca = PCA()
    pca.fit(cov_matrix)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
    print(f"Number of components selected: {num_components} out of {cov_matrix.shape[0]}")
    reduced_cov_matrix = pca.components_[:num_components]
    standard_normal_samples = np.random.normal(size=(num_samples, num_components))
    simulated_data = standard_normal_samples @ reduced_cov_matrix
    elapsed_time = time.time() - start_time
    return simulated_data, elapsed_time


def compute_frobenius_norm(original_cov, simulated_data):
    simulated_cov = np.cov(simulated_data, rowvar=False)
    return norm(original_cov - simulated_cov, 'fro')


def plot_cumulative_variance(cov_matrix, pca_simulated_data, cholesky_simulated_data):
    # Perform PCA on all matrices
    pca_original = PCA().fit(cov_matrix)
    pca_pca_simulated = PCA().fit(np.cov(pca_simulated_data, rowvar=False))
    pca_cholesky_simulated = PCA().fit(np.cov(cholesky_simulated_data, rowvar=False))

    # Extract cumulative variance explained
    original_cumulative_variance = np.cumsum(pca_original.explained_variance_ratio_)
    pca_cumulative_variance = np.cumsum(pca_pca_simulated.explained_variance_ratio_)
    cholesky_cumulative_variance = np.cumsum(pca_cholesky_simulated.explained_variance_ratio_)

    # Plot cumulative variance explained
    plt.figure(figsize=(10, 6))
    plt.plot(original_cumulative_variance, label='Original Covariance', color='blue', linestyle='solid')
    plt.plot(pca_cumulative_variance, label='PCA Simulated Covariance', color='orange', linestyle='dashed')
    plt.plot(cholesky_cumulative_variance, label='Cholesky Simulated Covariance', color='green', linestyle='dotted')

    # Add labels and legend
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained by Eigenvalues')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv('problem6.csv')
    print(f"Original shape: {df.shape}")
    if df.shape[0] != df.shape[1]:
        if df.shape[0] + 1 == df.shape[1]:
            df = df.iloc[:, 1:]
            print(f"Fixed shape: {df.shape}")
    cov_matrix = df.values
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Final covariance matrix is still not square. Check CSV formatting.")
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    if np.any(np.linalg.eigvals(cov_matrix) <= 0):
        print("Warning: Covariance matrix is not positive definite, adding small perturbation.")
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    num_samples = 10000
    pca_simulated_data, _ = simulate_pca(cov_matrix, num_samples)
    cholesky_simulated_data = np.random.multivariate_normal(mean=np.zeros(cov_matrix.shape[0]), cov=cov_matrix,
                                                            size=num_samples)
    plot_cumulative_variance(cov_matrix, pca_simulated_data, cholesky_simulated_data)
