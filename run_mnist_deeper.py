import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import data
import dnn
import dnn.convolution

np.random.seed(168153852)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cslu', default='/Users/galer/data/cslu_kids')
    return parser.parse_args()

def main():
    args = get_args()

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # num_samples = 10000
    # X, y = X[0:num_samples], y[0:num_samples]

    dictionary = { label: n for n, label in enumerate(sorted(set(y))) }
    reverse_dictionary = list(dictionary.keys())

    y = np.array([dictionary[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(.8 * len(X)), test_size=int(.2 * len(X)))
    print(f'train size: {len(y_train)}, test size: {len(y_test)}')

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    cnn = dnn.layers.Sequential(
        dnn.layers.Reshape((1, 28, 28)),
        dnn.convolution.Convolution2d(kernel=(5, 5), in_channels=1, out_channels=16, pad='same'),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(5, 5), in_channels=16, out_channels=16),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(5, 5), in_channels=16, out_channels=32, stride=(1, 1), pad=(0,0)),
        dnn.activations.ReLU(),
        dnn.layers.Reshape((-1,)),
        dnn.layers.Linear(4608, 512),
        dnn.layers.Linear(512, len(dictionary)),
    )
    cross_entropy = dnn.loss.CrossEntropyLoss()

    epochs = 30
    batch_size = 100
    learn_rate = 0.0001
    decay = 0.9999

    train_iterator = data.prep.DataIterator(X_train, y_train, batch_size=batch_size)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    for e in range(epochs):
        os.makedirs('dumps', exist_ok=True)
        with open(f'dumps/mnist_{start_time}_epoch_{e}.dump', 'wb') as dump_file:
            pickle.dump(cnn, dump_file)

        epoch_loss = 0
        for b, (batch_X, batch_y) in enumerate(train_iterator()):
            y_hat = cnn(batch_X)
            df = pd.DataFrame(y_hat, columns=dictionary.keys())
            df['true'] = [reverse_dictionary[i] for i in batch_y]
            df['predicted'] = [reverse_dictionary[i] for i in np.argmax(y_hat, axis=1)]
            df = df.reindex(['true', 'predicted', *dictionary.keys()], axis=1)
            # print(df.T)
            batch_mae = np.mean(np.argmax(y_hat, axis=1) != batch_y)
            loss = cross_entropy(batch_y, y_hat)
            loss.backward(learn_rate=learn_rate)
            epoch_loss += loss
            batch_loss = loss / len(batch_y)
            loss_so_far = epoch_loss / (len(batch_y) + batch_size * (b))
            print(f'Epoch {e} | Batch {b} | MAE: {batch_mae:.3f} | learn rate: {learn_rate:.6f} | ' \
                  f'batch loss: {batch_loss:.5f} | epoch_loss: {loss_so_far:.5f}')
            learn_rate *= decay

        # train_y_hat = cnn(X_train)
        # train_mae = np.mean(np.argmax(train_y_hat, axis=1) != y_train)
        # print(f'Epoch {e} train MAE:  {train_mae}')

        test_y_hat = cnn(X_test)
        test_mae = np.mean(np.argmax(test_y_hat, axis=1) != y_test)
        test_loss = cross_entropy(y_test, test_y_hat)
        print(f'Epoch {e} | Test MAE: {test_mae:.3f} | test loss: {test_loss/len(y_test):.5f}')


if __name__ == '__main__':
    main()