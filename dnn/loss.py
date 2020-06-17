import numpy as np

from dnn.backprop import Function


class CrossEntropyLoss(Function):
    def forward(self, y_true, y_hat):
        NUMERICAL_STABILITY = 1e-13
        y_one_hot = self._one_hot(y_true, y_hat.shape)
        softmax = self._softmax(y_hat)
        loss = -np.sum(y_one_hot * np.log(softmax + NUMERICAL_STABILITY))
        return loss

    def _backward(self, y_true, y_hat, f_X, delta_current, update):
        y_one_hot = self._one_hot(y_true, y_hat.shape)
        dcost_dz = self._softmax(y_hat) - y_one_hot
        return dcost_dz, None, None

    def _softmax(self, x):
        normalized = (x - np.expand_dims(np.max(x, axis=1), 1))
        numerator = np.exp(normalized)
        return numerator / np.expand_dims(np.sum(numerator, axis=1), 1)

    def _one_hot(self, x, shape):
        one_hot = np.zeros(shape)
        for n, i in enumerate(x):
            one_hot[n, i] = 1
        return one_hot
