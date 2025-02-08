import pandas as pd
import numpy as np

# 读取 CSV 文件
file_path = "problem2.csv"  # 确保文件路径正确
df = pd.read_csv(file_path)

# 计算协方差矩阵
cov_matrix = df.cov()

# Higham 方法计算最近正半定矩阵
def higham_psd(matrix, tol=1e-10, max_iter=100):
    Y = matrix.copy()
    for _ in range(max_iter):
        Y = (Y + Y.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(Y)
        eigenvalues[eigenvalues < tol] = 0
        Y = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        if np.all(np.linalg.eigvals(Y) >= -tol):
            break
    return Y

higham_psd_matrix = higham_psd(cov_matrix)

# Rebenato & Jackel 方法计算最近正半定矩阵
def near_psd(matrix, epsilon=1e-10):
    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(np.maximum(s, epsilon)) @ V
    return (B + H) / 2

near_psd_matrix = near_psd(cov_matrix)

# 输出计算结果
labels = ['x1', 'x2', 'x3', 'x4', 'x5']
formatted_higham = pd.DataFrame(higham_psd_matrix, index=labels, columns=labels)
print("result of Higham method：")
print(formatted_higham)

print("\nresult of the Rebenato & Jackel method：")
print(near_psd_matrix)
