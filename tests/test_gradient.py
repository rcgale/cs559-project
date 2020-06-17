import sys

import pytest
import numpy as np

from dnn.loss import CrossEntropyLoss
from dnn.activations import Sigmoid, ReLU
from dnn.layers import Linear, Sequential, Flatten, Reshape
from dnn.convolution import Convolution2d, MaxPool2d

###################################################

TEST_CASES_WRT_X = [
    (Sigmoid(), (2, 3)),
    (ReLU(), (2, 3)),
    (Linear(3, 3), (2, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=1, out_channels=1, pad='same'), (2, 1, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=1, out_channels=2, pad='same'), (2, 1, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=2, out_channels=1, pad='same'), (2, 2, 3, 3)),
    (MaxPool2d(pool_size=(3,3)), (2, 2, 5, 5)),
]
@pytest.mark.parametrize('func, x_shape', TEST_CASES_WRT_X)
def test_gradient_wrt_x(func, x_shape):
    X = np.random.normal(size=x_shape)
    expected_gradient = _expected_gradient(func, X)
    f_X = func(X)
    dx, dw, db = f_X.backward()
    np.testing.assert_almost_equal(expected_gradient, dx[0], decimal=7)

###################################################

TEST_CASES_WRT_W = [
    (Linear(12, 24, use_bias=True), (2, 12)),
    (Convolution2d(kernel=(3, 3), in_channels=1, out_channels=1, pad='same'), (2, 1, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=1, out_channels=2, pad='same'), (2, 1, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=2, out_channels=1, pad='same'), (2, 2, 3, 3)),
]
@pytest.mark.parametrize('graph, x_shape', TEST_CASES_WRT_W)
def test_wrt_w(graph, x_shape):
    X = np.random.normal(size=x_shape)
    f_X = graph(X)

    def wrt_weights(w):
        graph.weights = w
        return graph(X)

    expected_gradient = _expected_gradient(wrt_weights, graph.weights)
    dy = np.ones_like(f_X)
    if dy.ndim == 3:
        dy /= dy.shape[1]
    dx, dw, db = f_X.backward(dy)
    np.testing.assert_almost_equal(expected_gradient, dw[0], decimal=7)

###################################################

TEST_CASES_WRT_B = [
    (Linear(12, 24, use_bias=True), (2, 12)),
    (Convolution2d(kernel=(3, 3), in_channels=1, out_channels=1, pad='same'), (2, 1, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=1, out_channels=2, pad='same'), (2, 1, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=2, out_channels=1, pad='same'), (2, 2, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=3, out_channels=4, pad='same'), (2, 3, 3, 3)),
    (Convolution2d(kernel=(3, 3), in_channels=4, out_channels=3, pad='same'), (2, 4, 3, 3)),
]
@pytest.mark.parametrize('graph, x_shape', TEST_CASES_WRT_B)
def test_wrt_b(graph, x_shape):
    X = np.random.normal(size=x_shape)
    f_X = graph(X)

    def wrt_bias(b):
        graph.bias = b
        return graph(X)

    expected_gradient = _expected_gradient(wrt_bias, graph.bias)
    dx, dw, db = f_X.backward()
    np.testing.assert_almost_equal(expected_gradient, db[0], decimal=7)

###################################################

TEST_CASES_LOSS = [
    (CrossEntropyLoss(), np.random.randint(low=0, high=11, size=(1)), (1, 12)),
]

@pytest.mark.parametrize('func, y_true, y_hat_shape', TEST_CASES_LOSS)
def test_loss(func, y_true, y_hat_shape):
    y_hat = np.random.normal(size=y_hat_shape)

    loss = func(y_true, y_hat)
    dx, dw, db = loss.backward()

    def wrt_y_hat(y_hat):
        return func(y_true, y_hat)

    expected_gradient = _expected_gradient(wrt_y_hat, y_hat)
    np.testing.assert_almost_equal(expected_gradient, dx[0], decimal=7)

###################################################

TEST_CASES_FULL_WRT_W = [
    (
            Sequential(
                Linear(20, 10), ReLU(),
                Linear(10, 10), ReLU(),
                Linear(10, 5), Sigmoid(),
            ),
            (2, 20),
            np.random.randint(low=0, high=5, size=2),
            CrossEntropyLoss(),
    ),
    (
            Sequential(
                Reshape((1, 10, 5)),
                Convolution2d(kernel=(3,3), stride=(3,3), in_channels=1, out_channels=2, pad='same'),
                Convolution2d(kernel=(3,3), stride=(3,3), in_channels=2, out_channels=1, pad='same'),
                Flatten(),
                Linear(50, 5),
                Sigmoid(),
            ),
            (3, 10, 5),
            np.random.randint(low=0, high=5, size=3),
            CrossEntropyLoss()
    ),
]

@pytest.mark.parametrize('graph, x_shape, y_true, cost_function', TEST_CASES_FULL_WRT_W)
def test_full_wrt_w(graph, x_shape, y_true, cost_function):
    errors = {}
    X = np.random.normal(size=x_shape)
    y_hat = graph(X)
    loss = cost_function(y_true, y_hat)
    dx, dw, db = loss.backward(update=False)

    for l, actual_gradient in enumerate(dw[:-1]):
        if not hasattr(graph.items[l], 'weights'):
            continue
        print()
        print(f'Testing layer {l}: {type(graph.items[l]).__name__}')

        original_weights = np.array(graph.items[l].weights)
        def wrt_weights(w):
            graph.items[l].weights = w
            y_hat = graph(X)
            loss = cost_function(y_true, y_hat)
            return loss

        expected_gradient = _expected_gradient(wrt_weights, original_weights)

        try:
            np.testing.assert_almost_equal(expected_gradient, actual_gradient, decimal=7)
        except AssertionError as e:
            errors[str(l)] = e

        graph.items[l].weights = original_weights

    for e in errors.values():
        print(e, file=sys.stderr)
    if len(errors):
        raise AssertionError(f'Layers {",".join(errors.keys())} have gradient issues')

###################################################

def _expected_gradient(func, parameters, small_step=1e-4):
    gradient = np.zeros_like(parameters)
    for idx, x in np.ndenumerate(parameters):
        parameters_copy = parameters.copy()
        parameters_copy[idx] += small_step
        right = np.array(func(parameters_copy))

        parameters_copy = parameters.copy()
        parameters_copy[idx] -= small_step
        left = np.array(func(parameters_copy))

        numerical_gradient = (right - left) / (2 * small_step)
        gradient[idx] = np.sum(numerical_gradient)

    return gradient

if __name__ == '__main__':
    pytest.main()