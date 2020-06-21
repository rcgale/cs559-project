import numpy as np

from dnn.backprop import Function


class Sigmoid(Function):
    def forward(self, X):
        return  1 / (1 + np.exp(-X))

    def _backward(self, X, f_X, dy):
        gradient = np.multiply(f_X, (1 - f_X))
        dx = dy * gradient
        return dx, None, None


class ReLU(Function):
    def forward(self, X):
        relu = np.array(X)
        relu[relu < 0] = 0
        return relu

    def _backward(self, X, f_X, dy):
        dx = np.zeros_like(f_X)
        dx[f_X > 0] = 1
        dx = dy * dx
        return dx, None, None