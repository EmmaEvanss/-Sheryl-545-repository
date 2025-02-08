import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('problem3.csv')

# Extract x1 and x2
x1 = data['x1']
x2 = data['x2']

# Step 1: Using the Conditional Distribution Formula
# Calculate mean vector and covariance matrix
mean_vector = np.array([x1.mean(), x2.mean()])
cov_matrix = np.cov(x1, x2, bias=False)  # Use standard sample covariance

# Verify if covariance matrix is positive definite
if np.all(np.linalg.eigvals(cov_matrix) > 0):
    print("Covariance matrix is positive definite.")
else:
    print("Covariance matrix is not positive definite. Adjust the data or check calculations.")

# Extract parameters from covariance matrix
mu_x1 = mean_vector[0]
mu_x2 = mean_vector[1]
sigma_x1x1 = cov_matrix[0, 0]
sigma_x2x2 = cov_matrix[1, 1]
sigma_x1x2 = cov_matrix[0, 1]

# Given X1 = 0.6
x1_given = 0.6
mu_x2_given_x1 = mu_x2 + (sigma_x1x2 / sigma_x1x1) * (x1_given - mu_x1)
sigma_x2_given_x1 = sigma_x2x2 - (sigma_x1x2 ** 2) / sigma_x1x1

# Step 2: Using OLS
# Reshape x1 for LinearRegression
x1_reshaped = x1.values.reshape(-1, 1)

# Fit OLS model
model = LinearRegression().fit(x1_reshaped, x2)
ols_prediction = model.predict(np.array([[x1_given]]))[0]

# Step 3: Simulation using Cholesky Decomposition
# Generate multivariate normal samples
num_samples = 50000000  # Further increase sample size to improve stability
samples = np.random.multivariate_normal(mean_vector, cov_matrix, size=num_samples)

# Extract x1 and x2 from samples
x1_samples = samples[:, 0]
x2_samples = samples[:, 1]

# Debug: Check range of x1_samples
print(f"x1_samples range: {x1_samples.min()} to {x1_samples.max()}")

# Plot histogram to visualize x1 distribution
plt.hist(x1_samples, bins=100, alpha=0.7, color='blue', label='x1_samples')
plt.axvline(x=x1_given, color='red', linestyle='--', label='x1_given')
plt.title('Histogram of x1_samples')
plt.legend()
plt.show()
plt.close()

# Validate generated samples
simulated_mean = np.mean(samples, axis=0)
simulated_cov = np.cov(samples, rowvar=False)
print("Simulated Mean:", simulated_mean)
print("Simulated Covariance Matrix:\n", simulated_cov)

# Directly generate conditional samples
conditional_samples = np.random.normal(
    loc=mu_x2_given_x1,
    scale=np.sqrt(sigma_x2_given_x1),
    size=500000
)

# Calculate empirical mean and variance of conditional_samples
empirical_mean_conditional = np.mean(conditional_samples)
empirical_variance_conditional = np.var(conditional_samples)

# Print results
print("Using Conditional Distribution Formula:")
print(f"Mean of X2 given X1=0.6: {mu_x2_given_x1}")
print(f"Variance of X2 given X1=0.6: {sigma_x2_given_x1}")

print("\nUsing OLS:")
print(f"Predicted X2 for X1=0.6: {ols_prediction}")

print("\nSimulation Results:")
print(f"Empirical Mean of X2 (Filtered): {empirical_mean_conditional}")
print(f"Empirical Variance of X2 (Filtered): {empirical_variance_conditional}")
