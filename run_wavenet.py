import argparse
import statistics

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import data
import dnn

import dnn.convolution

LONGEST_UTTERANCE = 89836 # frames
# LONGEST_UTTERANCE = 562 # windows

RANDOM_STATE = 168153852
np.random.seed(RANDOM_STATE)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cslu', default='/Users/galer/data/cslu_kids')
    return parser.parse_args()

def main():
    args = get_args()
    single_word_utterances = [
        u
        for u in data.cslu_kids.get_all(args.cslu, scripted=True, spontaneous=False)
        if ' ' not in u.transcript
            and u.transcript in ['one', 'two', 'three', 'four']
    ]

    X, y, dictionary = data.prep.get_X_y_waveform(single_word_utterances, max_length=LONGEST_UTTERANCE)
    reverse_dictionary = list(dictionary.keys())

    grouped_split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    (train_idx, test_idx), = grouped_split.split(X, y, groups=[u.speaker_id for u in single_word_utterances])
    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]

    # train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=len(X) - int(.2 * len(X)), test_size=int(.2 * len(X)))
    print(f'train size: {len(train_y)}, test size: {len(test_y)}')

    cnn = dnn.layers.Sequential(
        dnn.layers.Reshape((1, -1, 1)),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=1, out_channels=1, dilation=(1, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=1, out_channels=2, dilation=(2, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=2, out_channels=4, dilation=(4, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=4, out_channels=8, dilation=(8, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=8, out_channels=16, dilation=(16, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=16, out_channels=32, dilation=(32, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(64, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(128, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(256, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(512, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(1024, 1)),
        dnn.activations.ReLU(),

        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(1, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(2, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(4, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(8, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(16, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(32, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(64, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(128, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(256, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(512, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(1024, 1)),
        dnn.activations.ReLU(),

        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(1, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(2, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(4, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(8, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(16, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(32, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(64, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(128, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(256, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(512, 1)),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(2, 1), in_channels=32, out_channels=32, dilation=(1024, 1)),
        dnn.activations.ReLU(),

        dnn.layers.Flatten(),
        dnn.layers.Linear(84864, 1024),
        dnn.activations.ReLU(),
        dnn.layers.Linear(1024, 128),
        dnn.activations.ReLU(),
        dnn.layers.Linear(128, len(dictionary)),
    )
    cross_entropy = dnn.loss.CrossEntropyLoss()

    epochs = 20
    batch_size = 1
    learn_rate = 0.001
    decay = 0.9999

    train_iterator = data.prep.DataIterator(train_X, train_y, batch_size=batch_size)

    for e in range(epochs):
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
            print(f'Epoch {e} | Batch {b} MAE: {batch_mae:.3f} | learn rate: {learn_rate:.6f} | ' \
                  f'batch loss: {batch_loss:.5f} | epoch_loss: {loss_so_far:.5f}')
            learn_rate *= decay

        # train_y_hat = cnn(train_X)
        # train_mae = np.mean(np.argmax(train_y_hat, axis=1) != train_y)
        # print(f'Epoch {e} train MAE:  {train_mae}')

        test_y_hat = cnn(test_X)
        test_mae = np.mean(np.argmax(test_y_hat, axis=1) != test_y)
        print(f'Epoch {e} test MAE: {test_mae}')



if __name__ == '__main__':
    main()