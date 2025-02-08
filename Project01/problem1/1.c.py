import numpy as np
import pandas as pd
from scipy.stats import norm, t
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'problem1.csv'
data = pd.read_csv(file_path)

# 拟合正态分布
mu, std = norm.fit(data["X"])

# 拟合 t 分布
df, loc, scale = t.fit(data["X"])

# 生成 x 值用于绘制概率密度函数
x = np.linspace(data["X"].min(), data["X"].max(), 1000)

# 正态分布 PDF
pdf_normal = norm.pdf(x, mu, std)

# t 分布 PDF
pdf_t = t.pdf(x, df, loc, scale)

# 绘制直方图和 PDF
plt.figure(figsize=(10, 6))
plt.hist(data["X"], bins=30, density=True, alpha=0.6, color='gray', label='Data Histogram')
plt.plot(x, pdf_normal, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})', linewidth=2, color='aqua')
plt.plot(x, pdf_t, label=f'T Fit (df={df:.2f}, loc={loc:.2f}, scale={scale:.2f})', linewidth=2,color='pink')
plt.title("Comparison of Normal and T-Distributions")
plt.xlabel("X")
plt.ylabel("Density")
plt.legend()
plt.show()
