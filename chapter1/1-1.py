import numpy as np

x = np.array([1, 2, 3])
print(x.__class__)
print(x.shape)
print(x.ndim)

W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.shape)
print(W.ndim)

X = np.array([[0, 1, 2], [3, 4, 5]])
print(f"W+X \n{W + X}")
print(f"W*X \n {W * X}")

# 重み
W1 = np.random.randn(2, 4)
# バイアス
b1 = np.random.randn(4)
# 入力
x = np.random.randn(10, 2)
h = x @ W1 + b1

def kkk:

