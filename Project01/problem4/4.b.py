import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the data
file_path = "problem4.csv"
df = pd.read_csv(file_path)

# Extract the first column to use as the basis for simulation (assuming it's numeric data)
data = df.iloc[:, 0]

# Normalize the data to avoid explosive AR processes
normalized_data = (data - data.mean()) / data.std()

# Define AR(1), AR(2), AR(3) parameters
ar1_params = np.array([1, -0.7])  # AR(1): phi1 = -0.7
ar2_params = np.array([1, -0.7, 0.3])  # AR(2): phi1 = -0.7, phi2 = 0.3
ar3_params = np.array([1, -0.7, 0.3, -0.2])  # AR(3): phi1 = -0.7, phi2 = 0.3, phi3 = -0.2

# Simulate AR processes based on the first column of the data
np.random.seed(42)
n_samples = len(normalized_data)

ar1_process = ArmaProcess(ar=ar1_params, ma=[1]).generate_sample(nsample=n_samples)
ar2_process = ArmaProcess(ar=ar2_params, ma=[1]).generate_sample(nsample=n_samples)
ar3_process = ArmaProcess(ar=ar3_params, ma=[1]).generate_sample(nsample=n_samples)

# Plot ACF and PACF for each AR process
fig, axes = plt.subplots(3, 2, figsize=(12, 9))

# AR(1) ACF & PACF
plot_acf(ar1_process, ax=axes[0, 0], lags=20, title="AR(1) ACF",color="violet", zero=False)
plot_pacf(ar1_process, ax=axes[0, 1], lags=20, title="AR(1) PACF",color="violet", zero=False)

# AR(2) ACF & PACF
plot_acf(ar2_process, ax=axes[1, 0], lags=20, title="AR(2) ACF",color="violet", zero=False)
plot_pacf(ar2_process, ax=axes[1, 1], lags=20, title="AR(2) PACF",color="violet", zero=False)

# AR(3) ACF & PACF
plot_acf(ar3_process, ax=axes[2, 0], lags=20, title="AR(3) ACF",color="violet", zero=False)
plot_pacf(ar3_process, ax=axes[2, 1], lags=20, title="AR(3) PACF",color="violet", zero=False)

plt.tight_layout()
plt.show()
