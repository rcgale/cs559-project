import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit

import data
import dnn
# LONGEST_UTTERANCE = 89836 # frames
import dnn.convolution
import dnn.train

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
    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]

    scaler = sklearn.preprocessing.StandardScaler()
    train_X = scaler.fit_transform(train_X.reshape((len(train_X), -1))).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape((len(test_X), -1))).reshape(test_X.shape)


    # train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=len(X) - int(.2 * len(X)), test_size=int(.2 * len(X)))
    print(f'train size: {len(train_y)}, test size: {len(test_y)}')

    cnn = dnn.layers.Sequential(
        dnn.layers.Reshape((1, -1, 26)),
        dnn.convolution.Convolution2d(kernel=(11, 25), in_channels=1, out_channels=32, stride=(2, 2), pad=(0, 'same')),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(11, 13), in_channels=32, out_channels=32, stride=(2, 2), pad=(0, 'same')),
        dnn.activations.ReLU(),
        dnn.convolution.Convolution2d(kernel=(11, 13), in_channels=32, out_channels=32, stride=(2, 2), pad=(0, 'same')),
        dnn.activations.ReLU(),
        dnn.layers.Flatten(),
        dnn.layers.Linear(27200, 100),
        dnn.layers.Linear(100, len(dictionary)),
    )
    cross_entropy = dnn.loss.CrossEntropyLoss()

    epochs = 20
    batch_size = 10
    learn_rate = 0.0001
    decay = 0.9999

    train_iterator = dnn.train.DataIterator(train_X, train_y, batch_size=batch_size)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    for e in range(epochs):
        os.makedirs('dumps', exist_ok=True)
        with open(f'dumps/cslu_tenwords_deeper_{start_time}_epoch_{e}.dump', 'wb') as dump_file:
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
            epoch_loss += np.array(loss)
            batch_loss = np.array(loss) / len(batch_y)
            loss_so_far = epoch_loss / (len(batch_y) + batch_size * (b))
            print(f'Epoch {e} | Batch {b} | MAE: {batch_mae:.3f} | learn rate: {learn_rate:.6f} | '
                  f'batch loss: {batch_loss:.5f} | epoch_loss: {loss_so_far:.5f}')
            learn_rate *= decay

        # train_y_hat = cnn(train_X)
        # train_mae = np.mean(np.argmax(train_y_hat, axis=1) != train_y)
        # print(f'Epoch {e} train MAE:  {train_mae}')

        test_y_hat = cnn(test_X)
        test_mae = np.mean(np.argmax(test_y_hat, axis=1) != test_y)
        test_loss = cross_entropy(test_y, test_y_hat)
        print(f'Epoch {e} | Test MAE: {test_mae:.3f} | test loss: {test_loss/len(test_y):.5f}')

        if test_loss > previous_test_loss:
            early_stop_count += 1
            if early_stop_count == 3:
                print ('Early stopping')
                break
        else:
            early_stop_count = 0
        previous_test_loss = np.array(test_loss)



if __name__ == '__main__':
    main()