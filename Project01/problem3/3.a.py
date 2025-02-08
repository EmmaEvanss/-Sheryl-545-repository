import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal


# 读取数据
file_path = "problem3.csv"
df = pd.read_csv(file_path)

# 计算均值向量和协方差矩阵
mean_vector = df.mean().values  # 计算每列的均值
cov_matrix = df.cov().values  # 计算协方差矩阵

# 拟合多元正态分布
multivariate_dist = multivariate_normal(mean=mean_vector, cov=cov_matrix)

# 创建网格数据用于绘制概率密度等高线
x, y = np.meshgrid(
    np.linspace(df.iloc[:, 0].min() - 0.5, df.iloc[:, 0].max() + 0.5, 100),
    np.linspace(df.iloc[:, 1].min() - 0.5, df.iloc[:, 1].max() + 0.5, 100)
)
pos = np.dstack((x, y))  # 组合坐标点

# 计算多元正态分布的概率密度
pdf_values = multivariate_dist.pdf(pos)

# 绘制数据散点图和等高线
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], label="Data Points", color='blue', alpha=0.6)
plt.contour(x, y, pdf_values, levels=10, cmap="Reds")  # 绘制等高线
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Multivariate Normal Fit")
plt.legend()
plt.grid()

# 显示图像
plt.show()
# 输出结果
print("mean vector:")
print(mean_vector)
print("\ncovariace matrix:")
print(cov_matrix)
