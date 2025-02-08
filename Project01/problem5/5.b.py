import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def exponential_weighted_covariance(data, lambd):
    """
    Calculate the exponentially weighted covariance matrix.
    Parameters:
        data: ndarray
            The input data where rows are observations and columns are variables.
        lambd: float
            The exponential decay factor (0 < lambd < 1).
    Returns:
        cov_matrix: ndarray
            The exponentially weighted covariance matrix.
    """
    T, N = data.shape
    weights = np.array([lambd ** (T - t - 1) for t in range(T)])
    weights /= weights.sum()  # Normalize weights

    mean_adjusted_data = data - np.average(data, axis=0, weights=weights)
    weighted_data = mean_adjusted_data * np.sqrt(weights)[:, np.newaxis]
    cov_matrix = np.dot(weighted_data.T, weighted_data)
    return cov_matrix


# Read data from problem5.csv
file_path = "DailyReturn.csv"
df = pd.read_csv(file_path)

# Extract numerical data (assuming the first column is not numerical data)
returns_data = df.iloc[:, 1:].values  # Skip the first column if it's a non-numerical index

# Normalize the data
returns_data = returns_data - np.mean(returns_data, axis=0)

# Define λ values in (0,1)
lambda_values = np.linspace(0.01, 0.99, 10)

# Store cumulative variance explained for each λ
cumulative_variance_dict = {}

# Loop over λ values and calculate PCA
for lambd in lambda_values:
    # Compute exponentially weighted covariance matrix
    ew_cov_matrix = exponential_weighted_covariance(returns_data, lambd)

    # Perform PCA on the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(ew_cov_matrix)  # Eigen decomposition
    eigvals = eigvals[::-1]  # Sort eigenvalues in descending order

    # Compute cumulative variance explained
    total_variance = np.sum(eigvals)
    cumulative_variance = np.cumsum(eigvals) / total_variance
    cumulative_variance_dict[lambd] = cumulative_variance

# Plot cumulative variance explained for different λ values
plt.figure(figsize=(10, 6))
for lambd, cumulative_variance in cumulative_variance_dict.items():
    plt.plot(cumulative_variance, label=f"λ={lambd:.2f}")

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Cumulative Variance Explained by PCA with Exponentially Weighted Covariance")
plt.legend()
plt.grid()
plt.show()
