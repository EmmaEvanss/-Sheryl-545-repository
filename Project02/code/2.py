import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib
matplotlib.use('Agg')  # 解决 Windows GUI 兼容问题
import matplotlib.pyplot as plt

# 读取数据
file_path = "code/DailyPrices.csv"  # 请确保文件路径正确
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# 定义投资组合
portfolio_weights = {'SPY': 100, 'AAPL': 200, 'EQIX': 150}

# 获取最新价格（假设1/3/2025是最后一行数据）
latest_prices = df.iloc[-1]

# 计算投资组合当前价值
portfolio_value = sum(latest_prices[ticker] * portfolio_weights[ticker] for ticker in portfolio_weights)
print(f"Current Portfolio Value: ${portfolio_value:.2f}")

# 计算算术收益率
returns = df.pct_change().dropna()
returns -= returns.mean()
returns = returns[['SPY', 'AAPL', 'EQIX']]

# 计算协方差矩阵（指数加权，lambda=0.97）
lambda_ = 0.97
ewma_cov = returns.ewm(span=(2 / (1 - lambda_)) - 1).cov().iloc[-len(returns.columns):]

# 计算投资组合的收益率（加权求和）
weights_vector = np.array([portfolio_weights[ticker] * latest_prices[ticker] / portfolio_value for ticker in ['SPY', 'AAPL', 'EQIX']])
portfolio_return = returns @ weights_vector

# 处理 portfolio_return 为空的情况
if portfolio_return.isnull().all():
    print("Error: portfolio_return is empty or contains only NaN values.")
    exit()

# 计算VaR和ES
alpha = 0.05
portfolio_std = np.sqrt(weights_vector.T @ ewma_cov @ weights_vector)

# (a) 正态分布 VaR & ES
VaR_norm = -norm.ppf(alpha) * portfolio_std
ES_norm = -portfolio_std * norm.pdf(norm.ppf(alpha)) / alpha

# (b) t 分布 Copula VaR & ES
df_t = 6
VaR_t = -t.ppf(alpha, df_t) * portfolio_std
ES_t = -portfolio_std * (t.pdf(t.ppf(alpha, df_t), df_t) * (df_t + t.ppf(alpha, df_t) ** 2) / (df_t - 1)) / alpha

# (c) 历史模拟法 VaR & ES
VaR_hist = -np.percentile(portfolio_return, alpha * 100)
ES_hist = -portfolio_return[portfolio_return <= -VaR_hist].mean()

# 输出结果
print(f"\nVaR & ES at 5% alpha level:")
print(f"Normal VaR: ${VaR_norm:.2f}, ES: ${ES_norm:.2f}")
print(f"T-distribution VaR: ${VaR_t:.2f}, ES: ${ES_t:.2f}")
print(f"Historical Simulation VaR: ${VaR_hist:.2f}, ES: ${ES_hist:.2f}")

# 画图
plt.hist(portfolio_return, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(-VaR_norm, color='r', linestyle='dashed', label="VaR (Normal)")
plt.axvline(-VaR_t, color='g', linestyle='dashed', label="VaR (T)")
plt.axvline(-VaR_hist, color='orange', linestyle='dashed', label="VaR (Historical)")
plt.legend()
plt.title("Portfolio Return Distribution with VaR")
plt.savefig("portfolio_return_distribution.png")  # 解决 plt.show() 报错
print("Histogram saved as portfolio_return_distribution.png")
