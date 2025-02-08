import pandas as pd

file_path = "problem2.csv"
df = pd.read_csv(file_path)

overlapping_data_cov_matrix = df.dropna().cov()

print("Covariance Matrix with Overlapping Data:")
print(overlapping_data_cov_matrix)
