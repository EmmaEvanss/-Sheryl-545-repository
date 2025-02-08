import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 读取数据
file_path = "problem4.csv"
df = pd.read_csv(file_path)

# 设置 MA(1), MA(2), MA(3) 过程的参数
ma1_params = np.array([1, -0.5])  # MA(1): θ1 = -0.5
ma2_params = np.array([1, -0.5, 0.3])  # MA(2): θ1 = -0.5, θ2 = 0.3
ma3_params = np.array([1, -0.5, 0.3, -0.2])  # MA(3): θ1 = -0.5, θ2 = 0.3, θ3 = -0.2

# 生成随机噪声
np.random.seed(42)
n_samples = 200

# 生成 MA(1), MA(2), MA(3) 过程的时间序列
ma1_process = ArmaProcess(ma=np.r_[1, ma1_params[1:]]).generate_sample(nsample=n_samples)
ma2_process = ArmaProcess(ma=np.r_[1, ma2_params[1:]]).generate_sample(nsample=n_samples)
ma3_process = ArmaProcess(ma=np.r_[1, ma3_params[1:]]).generate_sample(nsample=n_samples)

# 绘制 ACF 和 PACF 图
fig, axes = plt.subplots(3, 2, figsize=(12, 9))

# MA(1) ACF & PACF
plot_acf(ma1_process, ax=axes[0, 0], lags=20, title="MA(1) ACF", color="violet", zero=False)
plot_pacf(ma1_process, ax=axes[0, 1], lags=20, title="MA(1) PACF", color="violet", zero=False)

# MA(2) ACF & PACF
plot_acf(ma2_process, ax=axes[1, 0], lags=20, title="MA(2) ACF", color="violet", zero=False)
plot_pacf(ma2_process, ax=axes[1, 1], lags=20, title="MA(2) PACF", color="violet", zero=False)

# MA(3) ACF & PACF
plot_acf(ma3_process, ax=axes[2, 0], lags=20, title="MA(3) ACF", color="violet", zero=False)
plot_pacf(ma3_process, ax=axes[2, 1], lags=20, title="MA(3) PACF", color="violet", zero=False)


plt.tight_layout()
plt.show()
