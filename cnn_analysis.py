import argparse
import os
import pickle
import random

from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml

import dnn
from dnn.layers import Reshape


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='dumps/mnist_cnn_20200617_103935_epoch_21.dump')
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.file, 'rb') as f:
        model = pickle.load(f)
    conv_layers = [i for i in model.items if type(i) == dnn.convolution.Convolution2d ]
    exp = os.path.basename(args.file).replace('.dump', '')
    dir = f'analysis/{exp}'
    os.makedirs(dir, exist_ok=True)

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    examples = np.vstack([
        next(X for X, y in zip(X, y) if y == label)
        for label in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    ])
    examples_processed = Reshape((1, 28, 28))(examples)

    for n, conv_layer in enumerate(conv_layers):
        layer_weights = np.ones((4 * conv_layer.weights.shape[0], 4 * conv_layer.weights.shape[1]))
        examples_processed = conv_layer(examples_processed)
        ex_n, ex_c, ex_h, ex_w = examples_processed.shape
        layer_processed = np.ones((ex_h * ex_c, ex_w * ex_n))
        for co in range(conv_layer.weights.shape[0]):
            for ex in range(len(examples_processed)):
                layer_processed[ex_h*co:ex_h*(co+1), ex_w*ex:ex_w*(ex+1)] = normalize(examples_processed[ex, co])
                save(normalize(examples_processed[ex, co]), os.path.join(dir, f'example_{n}_{co}_{ex}_.png'))
            # for ci in range(conv_layer.weights.shape[1]):
            #     layer_weights[4*co:4*(co+1), 4*ci:4*(ci+1)] = normalize(conv_layer.weights[co, ci]).T
        print(layer_processed.shape)
        save(layer_processed, os.path.join(dir, f'example_{n}.png'))
        save(layer_weights, os.path.join(dir, f'weights_{n}.png'))

def normalize(pixels):
    pixels = (pixels - pixels.mean()) / (pixels.std() * 4) + 0.5
    return np.clip(255 * pixels, 0, 255)

def save(pixels, filename):
    im = Image.new('L', pixels.shape)
    im.putdata(pixels.flatten())
    im.save(filename)


if __name__ == '__main__':
    main()