import argparse

import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit

import data
import dnn
from dnn.train import do_train

# LONGEST_UTTERANCE = 89836 # frames
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
            and u.transcript in ['one', 'two', 'three', 'four']
    ]

    X, y, dictionary = data.prep.get_X_y(single_word_utterances, max_length=LONGEST_UTTERANCE, n_features=26)

    grouped_split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    (train_idx, test_idx), = grouped_split.split(X, y, groups=[u.speaker_id for u in single_word_utterances])
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape((len(X_train), -1))).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape((len(X_test), -1))).reshape(X_test.shape)

    print(f'train size: {len(y_train)}, test size: {len(y_test)}')

    cnn = dnn.layers.Sequential(
        dnn.layers.Reshape((1, -1, 26)),
        dnn.convolution.Convolution2d(kernel=(11, 25), in_channels=1, out_channels=32, stride=(2, 2), pad=(0, 'same')),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(11, 13), in_channels=32, out_channels=32, stride=(2, 2), pad=(0, 'same')),
        dnn.activations.ReLU(),
        dnn.layers.Flatten(),
        dnn.layers.Linear(116960, len(dictionary)),
    )

    do_train(
        model=cnn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cost_function=dnn.loss.CrossEntropyLoss(),
        epochs=30,
        batch_size=40,
        learn_rate=0.0001,
        decay=0.9999,
        exp_name='fourwords'
    )


if __name__ == '__main__':
    main()