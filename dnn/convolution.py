import numpy as np

from dnn.backprop import Function


class Convolution2d(Function):
    def __init__(self, kernel, stride=(1, 1), in_channels=1, out_channels=1, pad=(0, 0), weights=None, bias=None, he_init=True):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = weights
        self.pad = pad
        if weights is None and not he_init:
            self.weights = np.random.normal(
                scale=0.01,
                size=(out_channels, in_channels, *kernel)
            )
        if bias is None and not he_init:
            self.bias = np.random.normal(scale=0.01, size=(out_channels))

    def forward(self, X):
        self._he_init(X)

        P0, P1 = self._get_pad(X)
        S0, S1 = self.stride
        CO, CI, K0, K1 = self.weights.shape
        N, CI, X0, X1 = X.shape

        X_padded = np.pad(X, [(0,0), (0,0), (P0, P0), (P1, P1)])

        out = np.zeros((N, CO, (1 + (X0 + 2 * P0 - K0) // S0), (1 + (X1 + 2 * P1 - K1) // S1)))

        # Optimization: not iterating over N or channels so numpy has some room to do parallel computations
        for (x0, x1), _ in np.ndenumerate(out[0, 0]):
            x0_slice = slice((x0 * S0), (x0 * S0 + K0))
            x1_slice = slice((x1 * S1), (x1 * S1 + K1))
            X_window = np.expand_dims(X_padded[:, :, x0_slice, x1_slice], 1)
            out[:, :, x0, x1] += np.sum(np.multiply(X_window, self.weights), axis=(2, 3, 4)) + self.in_channels * self.bias[:]
        return out

    def _backward(self, X, f_X, dy, update):
        P0, P1 = self._get_pad(X)
        CO, CI, K0, K1 = self.weights.shape
        S0, S1 = self.stride

        X_padded = np.pad(X, [(0,0), (0,0), (P0, P0), (P1, P1)])

        # We want to return three gradients, w.r.t. X, weights, and bias
        dx = np.zeros_like(X_padded)
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)

        db += np.sum(dy, axis=(0, 2, 3)) * self.in_channels

        # Optimization: not iterating over N so numpy has some room to do threaded computations
        for (x0, x1), _ in np.ndenumerate(dy[0, 0]):
            x0_slice = slice((x0 * S0), (x0 * S0 + K0)) # Start and stop points for convolutional window
            x1_slice = slice((x1 * S1), (x1 * S1 + K1)) # Start and stop points for convolutional window

            dy_expanded = np.expand_dims(dy[:, :, x0:x0+1, x1:x1+1], 2) # Adding a channel in dimension
            dx[:, :, x0_slice, x1_slice] += np.sum(np.multiply(self.weights[:, :], dy_expanded), axis=(1))

            X_window = np.expand_dims(X_padded[:, :, x0_slice, x1_slice], 1)
            dw[:, :] += np.sum(np.multiply(X_window, dy_expanded), axis=(0))

        # Trim off padding from dx
        dx = dx[:, :, P0:dx.shape[2]-P0, P1:dx.shape[3]-P1]

        return dx, dw, db

    def _get_pad(self, X):
        if self.pad == 'same' or self.pad[0] == 'same':
            kernel_span_0 = self.weights.shape[2] + (self.weights.shape[2] - 1)
            pad_0 = (self.stride[0] * (X.shape[2] - 1) - X.shape[2] + kernel_span_0) // 2
        else:
            pad_0 = self.pad[0]

        if self.pad == 'same' or self.pad[1] == 'same':
            kernel_span_1 = self.weights.shape[3] + (self.weights.shape[3] - 1)
            pad_1 = (self.stride[1] * (X.shape[3] - 1) - X.shape[3] + kernel_span_1) // 2
        else:
            pad_1 = self.pad[1]

        return pad_0, pad_1

    def _he_init(self, X):
        if self.weights is not None:
            return
        shape = (self.out_channels, self.in_channels, *self.kernel)
        scale = np.sqrt(2 / np.product(X.shape[1:]))
        self.weights = np.random.normal(scale=scale, size=shape)
        self.bias = np.random.normal(scale=scale, size=(self.out_channels,))


class MaxPool2d(Function):
    def __init__(self, pool_size, stride=(1,1)):
        super().__init__()
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, X):
        out_height = int(1 + (X.shape[2] - self.pool_size[0]) / self.stride[0])
        out_width = int(1 + (X.shape[3] - self.pool_size[1]) / self.stride[1])
        out = np.zeros([X.shape[0], X.shape[1], out_height, out_width])
        for (x1, x2), _ in np.ndenumerate(out[0, 0]):
            x1_slice, x2_slice = _slices(x1, x2, self.pool_size[0], self.pool_size[1], self.stride)
            out[x1, x2] = np.max(X[n, c, x1_slice, x2_slice])
        return out

    def _backward(self, X, f_X, dy, update):
        dx = np.zeros_like(X)
        for (n, c, x1, x2), value in np.ndenumerate(dy):
            x1_slice, x2_slice = _slices(x1, x2, self.pool_size[0], self.pool_size[1], self.stride)
            x1_idx, x2_idx = np.unravel_index(X[n, c, x1_slice, x2_slice].argmax(), X[n, c, x1_slice, x2_slice].shape)
            dx[n, c, x1_idx + x1 * self.stride[0], x2_idx + x2 * self.stride[1]] += value
        return dx, None, None


def _slices(x1, x2, h, w, stride):
    x1_begin = x1 * stride[0]
    x1_end = x1_begin + h
    x2_begin = x2 * stride[1]
    x2_end = x2_begin + w
    return slice(x1_begin, x1_end), slice(x2_begin, x2_end)
