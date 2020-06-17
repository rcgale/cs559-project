import argparse
import os
import pickle
from PIL import Image
import numpy as np

import dnn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='dumps/mnist_20200614_195512_epoch_12.dump')
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.file, 'rb') as f:
        model = pickle.load(f)
    conv_layers = [i for i in model.items if type(i) == dnn.convolution.Convolution2d ]
    exp = os.path.basename(args.file).replace('.dump', '')
    dir = f'analysis/{exp}'
    os.makedirs(dir, exist_ok=True)

    for conv_layer in conv_layers:
        for co in range(conv_layer.weights.shape[0]):
            for ci in range(conv_layer.weights.shape[1]):
                png_file = f'{co}_{ci}.png'
                pixels = conv_layer.weights[co, ci]
                pixels = (pixels - pixels.mean()) / (pixels.std() * 4) + 0.5
                pixels = np.clip(255 * pixels, 0, 255)
                im = Image.new('L', conv_layer.weights.shape[2:])
                im.putdata(pixels.flatten())
                im.save(os.path.join(dir, png_file))

    pass


if __name__ == '__main__':
    main()