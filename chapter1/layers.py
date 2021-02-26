import numpy as np

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
        dx = dout * (1.0 - self.out) * self.out
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