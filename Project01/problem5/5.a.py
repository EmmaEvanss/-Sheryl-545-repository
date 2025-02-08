import numpy as np
import pandas as pd

def calculate_ew_covariance(data, alpha):
    """
    Calculate exponentially weighted covariance matrix.

    Parameters:
        data (pd.DataFrame): DataFrame where each column is a time series.
        alpha (float): Smoothing parameter, 0 < alpha <= 1.

    Returns:
        np.ndarray: Exponentially weighted covariance matrix.
    """
    weights = np.exp(np.linspace(-1, 0, len(data)))  # Generate weights
    weights /= weights.sum()  # Normalize weights

    mean_values = data.mean(axis=0)
    centered_data = data - mean_values

    weighted_cov = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            weighted_cov[i, j] = np.sum(weights * centered_data.iloc[:, i] * centered_data.iloc[:, j])

    return weighted_cov

# Load the data
data = pd.read_csv('DailyReturn.csv')
data = data.set_index(data.columns[0])  # Set first column as index

# Example: Compute exponentially weighted covariance matrix
alpha = 0.94
ew_cov_matrix = calculate_ew_covariance(data, alpha)

print("Exponentially Weighted Covariance Matrix:")
print(ew_cov_matrix)

# Verification step: Compare against package implementation (e.g., pandas.ewm)
pd_ewm_cov = data.ewm(alpha=alpha).cov(pairwise=True).iloc[-data.shape[1]:]
print("Verification with pandas ewm:")
print(pd_ewm_cov)
