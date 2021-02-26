import numpy as np

# 1.3.4.3 Repeatノード
D, N = 8, 7
x = np.random.randn(1, D) # 入力
y = np.repeat(x, N, axis=0) # forward

dy = np.random.randn(N, D) # 仮の勾配
dx = np.sum(dy, axis=0, keepdims=True) # backward

# 1.3.4.4 Sumノード
x = np.random.randn(N, D) # 入力
y = np.sum(x, axis=0, keepdims=True) # forward

dy = np.random.randn(1, D) # 仮の勾配
dx = np.repeat(dy, N, axis=0) # backward

# 1.3.4.5 MatMulノード
class MatMul:
	def __init__(self, W):
		self.params = [W]
		self.grads = [np.zeros_like(W)]
		self.x = None

	def forward(self, x):
	    W, = self.params
	    out = x@W
	    self.x = x
	    return out

	def backward(self, dout):
	    W, = self.params
	    dx = dout@W.T
	    dW = x.T@dout
	    # a=b はaの指すメモリ位置がbになる
	    # a[...]=b はaの指すメモリの位置は変わらずにbをコピーする
	    self.grads[0][...] = dW
	    return dx

# 1.3.5.1 Sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.paramas, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - selfout) * self.out
        return dx

# 1.3.5.2 Affineレイヤ
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = x@W + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = dout@W.T
        dW = self.x.T@dout
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx