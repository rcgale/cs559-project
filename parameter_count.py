import numpy as np

from dnn.activations import Sigmoid, ReLU
from dnn.convolution import Convolution2d
from dnn.layers import Sequential, Reshape, Linear, Flatten

dnn = Sequential(
    Reshape((-1,)),
    Linear(784, 256),
    Sigmoid(),
    Linear(256, 10),
    #~200k
)

cnn = Sequential(
    Reshape((1, 28, 28)),
    Convolution2d(kernel=(3, 3), in_channels=1, out_channels=16, pad='same'),
    ReLU(),
    Convolution2d(kernel=(3, 3), in_channels=16, out_channels=32),
    ReLU(),
    Flatten(),
    Linear(25088, 10),
 #28*28+3*3*16+3*3*16*32+25088*10 = ~256k
)
primed = cnn(np.zeros((100,1,28,28)))

num_parameters = 0
for layer in dnn.items:
    if hasattr(layer, 'weights'):
        num_parameters += len(layer.weights.flatten())
    if hasattr(layer, 'bias'):
        num_parameters += len(layer.bias.flatten())
print(f'DNN: {num_parameters} parameters')

num_parameters = 0
for layer in cnn.items:
    if hasattr(layer, 'weights') and layer.weights is not None:
        num_parameters += len(layer.weights.flatten())
    if hasattr(layer, 'bias') and layer.bias is not None:
        num_parameters += len(layer.bias.flatten())
print(f'CNN: {num_parameters} parameters')
