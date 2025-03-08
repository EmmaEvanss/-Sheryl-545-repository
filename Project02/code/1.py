import pandas as pd
import numpy as np

# 读取数据
file_path = "code\DailyPrices.csv"  # 请确保文件路径正确
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# 计算算术收益率
arithmetic_returns = df.pct_change().dropna()

# 去均值，使均值为0
arithmetic_returns -= arithmetic_returns.mean()

# 计算标准差
arithmetic_std = arithmetic_returns.std()

# 显示最后5行和总标准差
print("Arithmetic Returns (last 5 rows):")
print(arithmetic_returns.tail())
print("\nTotal Standard Deviation:")
print(arithmetic_std)

# 计算对数收益率
log_returns = np.log(df / df.shift(1)).dropna()

# 去均值，使均值为0
log_returns -= log_returns.mean()

# 计算标准差
log_std = log_returns.std()

# 显示最后5行和总标准差
print("\nLog Returns (last 5 rows):")
print(log_returns.tail())
print("\nTotal Standard Deviation:")
print(log_std)
