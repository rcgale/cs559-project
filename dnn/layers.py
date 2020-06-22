import numpy as np

from dnn.backprop import Function


class Linear(Function):
    def __init__(self, input_size, output_size, weights=None, bias=None):
        super().__init__()
        self.weights = weights
        if weights is None:
            self.weights = np.random.normal(scale=1/(input_size+output_size), size=(input_size, output_size))
        if bias is None:
            self.bias = np.random.normal(scale=1/(input_size+output_size), size=(1, output_size))


    def forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def _backward(self, X, f_X, dy):
        dx = np.dot(dy, self.weights.T)
        dw = np.dot(X.T, dy)
        db = np.sum(dy, axis=(0)).reshape(self.bias.shape)
        return dx, dw, db


class Sequential(object):
    def __init__(self, *items):
        super().__init__()
        self.items = items

    def __call__(self, X):
        for module in self.items:
            X = module(X)
        return X


class Flatten(Function):
    def forward(self, X):
        return X.reshape((X.shape[0], -1))

    def _backward(self, X, f_X, delta_current):
        d = delta_current.reshape(X.shape)
        return d, d, None


class Reshape(Function):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, X):
        return X.reshape((X.shape[0], *self.shape))

    def _backward(self, X, f_X, delta_current):
        d = delta_current.reshape(X.shape)
        return d, d, None
