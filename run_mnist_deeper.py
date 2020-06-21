import argparse

import numpy as np
import sklearn.preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import dnn
import dnn.convolution
from dnn.train import do_train

np.random.seed(168153852)

def main():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    dictionary = { label: n for n, label in enumerate(sorted(set(y))) }

    y = np.array([dictionary[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(.8 * len(X)), test_size=int(.2 * len(X)))
    print(f'train size: {len(y_train)}, test size: {len(y_test)}')

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    cnn = dnn.layers.Sequential(
        dnn.layers.Reshape((1, 28, 28)),
        dnn.convolution.Convolution2d(kernel=(3, 3), in_channels=1, out_channels=16, pad='same'),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(3, 3), in_channels=16, out_channels=32),
        dnn.activations.ReLU(),
        dnn.layers.Reshape((-1,)),
        dnn.layers.Linear(25088, len(dictionary)),
    )

    do_train(
        model=cnn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cost_function=dnn.loss.CrossEntropyLoss(),
        epochs=30,
        batch_size=100,
        learn_rate=0.001,
        decay=1.0,
        exp_name='mnist_playground'
    )


if __name__ == '__main__':
    main()