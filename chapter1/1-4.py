import numpy as np
import sys
sys.path.append("..")
from dataset import spiral
import matplotlib.pyplot as plt
from 1-4 import Affine, Sigmoid, SoftmaxWithLoss

def spiral_prot():
    x, t = spiral.load_data()
    print(f"x \n{x.shape}")
    print(f"t \n{t.shape}")

    # データ点のプロット
    N = 100
    CLS_NUM = 3
    markers = ["o", "x", "^"]
    for i in range(CLS_NUM):
        # 散布図のプロット
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
    plt.show()

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = npzeros(O)

        # レイヤの作成
        self.layers = [
            Affine(W1, b1),
            Sigmoid()
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 全ての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            



