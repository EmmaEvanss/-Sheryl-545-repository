import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('problem3.csv')

# Extract x1 and x2
x1 = data['x1']
x2 = data['x2']

# Step 1: Using the Conditional Distribution Formula
# Calculate mean vector and covariance matrix
mean_vector = np.array([x1.mean(), x2.mean()])
cov_matrix = np.cov(x1, x2)

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

# Print results
print("Using Conditional Distribution Formula:")
print(f"Mean of X2 given X1=0.6: {mu_x2_given_x1}")
print(f"Variance of X2 given X1=0.6: {sigma_x2_given_x1}")

print("\nUsing OLS:")
print(f"Predicted X2 for X1=0.6: {ols_prediction}")
