import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load the data
file_path = "problem4.csv"
df = pd.read_csv(file_path)

# Extract the first column of data (assuming it is the target time series)
data = df.iloc[:, 0]

# Define a function to compute AICc
def calculate_aicc(model_result, n, k):
    """Calculate the corrected Akaike Information Criterion (AICc)."""
    aic = model_result.aic
    return aic + (2 * k * (k + 1)) / (n - k - 1)

# Fit different AR and MA models and compare AICc
models = {
    "AR(1)": (1, 0, 0),
    "AR(2)": (2, 0, 0),
    "AR(3)": (3, 0, 0),
    "MA(1)": (0, 0, 1),
    "MA(2)": (0, 0, 2),
    "MA(3)": (0, 0, 3),
}

aicc_results = {}

# Fit models and store AICc values
for name, order in models.items():
    try:
        model = ARIMA(data, order=order)
        result = model.fit()
        aicc_value = calculate_aicc(result, len(data), sum(order))
        aicc_results[name] = aicc_value
        print(f"{name}: AICc = {aicc_value:.2f}")
    except Exception as e:
        print(f"Model {name} failed: {e}")

# Find the best model
best_model = min(aicc_results, key=aicc_results.get)
print(f"\nBest fitting model: {best_model} with AICc = {aicc_results[best_model]:.2f}")
