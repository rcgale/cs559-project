import argparse

import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit

import data
import dnn
# LONGEST_UTTERANCE = 89836 # frames
from dnn.activations import ReLU
from dnn.convolution import Convolution2d
from dnn.layers import Linear, Flatten, Reshape
from dnn.train import do_train

LONGEST_UTTERANCE = 562 # windows

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
            and u.transcript in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    ]

    X, y, dictionary = data.prep.get_X_y(single_word_utterances, max_length=LONGEST_UTTERANCE, n_features=26)
    reverse_dictionary = list(dictionary.keys())

    grouped_split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    (train_idx, test_idx), = grouped_split.split(X, y, groups=[u.speaker_id for u in single_word_utterances])
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape((len(X_train), -1))).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape((len(X_test), -1))).reshape(X_test.shape)


    # train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=len(X) - int(.2 * len(X)), test_size=int(.2 * len(X)))
    print(f'train size: {len(y_train)}, test size: {len(y_test)}')

    cnn = dnn.layers.Sequential(
        Reshape((1, -1, 26)),
        Convolution2d(kernel=(11, 25), in_channels=1, out_channels=32, stride=(2, 2), pad=(0, 'same')),
        ReLU(),
        Convolution2d(kernel=(11, 13), in_channels=32, out_channels=32, stride=(2, 2), pad=(0, 'same')),
        ReLU(),
        Flatten(),
        Linear(132096, len(dictionary)),
    )

    do_train(
        model=cnn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cost_function=dnn.loss.CrossEntropyLoss(),
        epochs=100,
        batch_size=40,
        learn_rate=0.0001,
        decay=0.9999,
        exp_name='tenwords'
    )




if __name__ == '__main__':
    main()