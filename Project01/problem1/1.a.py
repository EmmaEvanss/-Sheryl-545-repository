import pandas as pd
from scipy.stats import skew, kurtosis

# 读取CSV文件
file_path = 'problem1.csv'
data = pd.read_csv(file_path)

# 计算均值、方差、偏度和峰度
mean = data["X"].mean()
variance = data["X"].var()
skewness = skew(data["X"])
kurt = kurtosis(data["X"])

print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurt}")
