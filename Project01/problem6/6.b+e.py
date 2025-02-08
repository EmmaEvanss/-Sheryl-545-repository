import numpy as np
import time
from scipy.linalg import cholesky

# Load covariance matrix (replace with actual data)
cov_matrix = np.loadtxt("problem6.csv", delimiter=",", skiprows=1)

# Ensure positive definiteness
epsilon = 1e-6
cov_matrix_modified = cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon

# Cholesky decomposition
start_cholesky = time.time()
L = cholesky(cov_matrix_modified, lower=True)
z = np.random.randn(cov_matrix.shape[0], 10000)  # 10000 samples
samples_cholesky = L @ z
end_cholesky = time.time()
time_cholesky = end_cholesky - start_cholesky

# PCA with 75% variance
start_pca_75 = time.time()
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_modified)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

# Select components for 75% variance
cumulative_variance_75 = np.cumsum(eigenvalues) / np.sum(eigenvalues)
num_components_75 = np.searchsorted(cumulative_variance_75, 0.75) + 1

# Generate samples
selected_eigenvectors_75 = eigenvectors[:, :num_components_75]
selected_eigenvalues_75 = np.diag(np.sqrt(eigenvalues[:num_components_75]))
z_pca_75 = np.random.randn(10000, num_components_75)  # Ensure correct dimensions
samples_pca_75 = (selected_eigenvectors_75 @ selected_eigenvalues_75) @ z_pca_75.T

end_pca_75 = time.time()
time_pca_75 = end_pca_75 - start_pca_75

# Compute covariance of generated PCA samples
cov_pca_75 = np.cov(samples_pca_75, rowvar=True)  # Ensure covariance matches original structure

# Compute Frobenius norm to measure similarity to original covariance matrix
frobenius_norm_pca = np.linalg.norm(cov_pca_75 - cov_matrix_modified, 'fro')

# Print results
print(f"Cholesky Time: {time_cholesky:.6f} seconds")
print(f"PCA (75% variance) Time: {time_pca_75:.6f} seconds")
print(f"Frobenius Norm (PCA): {frobenius_norm_pca:.6f}")
