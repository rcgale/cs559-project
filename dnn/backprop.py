import multiprocessing
from multiprocessing import Pool

import numpy as np

MAX_PROCESSES = max(1, multiprocessing.cpu_count() - 1)


class Function(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        f_X = self.forward(*args)
        return BackpropWrapper(self, args, f_X)

    def forward(self, *args):
        raise Exception('Must implement forward()')

    def _backward(self, *args):
        raise Exception('Must implement backward()')


class BackpropWrapper(np.ndarray):
    def __init__(self, module, args, f_X):
        self.module = module
        self._args = args
        self._f_X = f_X

    def __new__(self, module, args, f_X):
        if isinstance(f_X, BackpropWrapper):
            return f_X
        if isinstance(f_X, np.ndarray):
            return f_X.view(self)
        arr = np.array(f_X)
        return np.ndarray.__new__(BackpropWrapper, shape=arr.shape, dtype=arr.dtype, buffer=arr)

    def detach(self):
        return np.array(self).copy()

    def backward(self, dy=None, update=True, learn_rate=0):
        if dy is None:
            dy = np.ones_like(self._f_X)
        dx, dw, db = self.module._backward(*self._args, self._f_X, dy, update)

        if update and hasattr(self.module, 'weights'):
            self.module.weights -= learn_rate * dw

        if update and hasattr(self.module, 'bias'):
            self.module.bias -= learn_rate * db

        dxs = []
        dws = []
        dbs = []
        for arg in self._args:
            if hasattr(arg, 'backward'):
                inner_dx, inner_dw, inner_db = arg.backward(dx, update, learn_rate)
                dxs.extend(inner_dx)
                dws.extend(inner_dw)
                dbs.extend(inner_db)
                break # Currently only support one input path for backpropagation
        dxs.append(np.array(dx))
        dws.append(np.array(dw))
        dbs.append(np.array(db))
        return dxs, dws, dbs
