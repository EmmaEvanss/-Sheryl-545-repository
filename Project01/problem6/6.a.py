import numpy as np
import pandas as pd
from numpy.linalg import cholesky, norm


# Load the covariance matrix from CSV file
def load_covariance_matrix(file_path):
    cov_matrix = pd.read_csv(file_path, skiprows=1, header=None).astype(float).values
    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Ensure symmetry
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6  # Ensure positive definiteness
    return cov_matrix


# Generate multivariate normal samples using Cholesky decomposition
def cholesky_simulation(cov_matrix, num_samples=10000):
    L = cholesky(cov_matrix)
    z = np.random.randn(num_samples, cov_matrix.shape[0])
    samples = z @ L.T
    return samples


# Compute the covariance of the generated samples
def compute_covariance(samples):
    return np.cov(samples, rowvar=False)


# Compute the Frobenius norm between two matrices
def compute_frobenius_norm(mat1, mat2):
    return norm(mat1 - mat2, 'fro')


if __name__ == "__main__":
    file_path = "problem6.csv"  # Update the path if needed
    cov_matrix = load_covariance_matrix(file_path)
    cholesky_samples = cholesky_simulation(cov_matrix)
    cov_cholesky = compute_covariance(cholesky_samples)
    frobenius_norm_cholesky = compute_frobenius_norm(cov_cholesky, cov_matrix)

    print(f"Frobenius Norm (Cholesky): {frobenius_norm_cholesky}")
