import python_speech_features

import data
import numpy as np


def get_X_y(utterances, max_length, n_features):
    dictionary = { word: i for i, word in enumerate(sorted(set(u.transcript for u in utterances))) }
    X = np.zeros((len(utterances), max_length, n_features))
    y = np.zeros((len(utterances)), dtype=np.int)
    min_start = max_length
    max_end = 0
    for n, utterance in enumerate(utterances):
        fs, audio = data.audio.read_sound_file(utterance.filename)

        features, energy = python_speech_features.fbank(
            audio,
            nfilt=n_features,
            samplerate=fs,
            winfunc=np.hamming,
        )

        start = (X.shape[1] - features.shape[0]) // 2
        end = start + features.shape[0]
        X[n, start:end, :] = features
        y[n] = dictionary[utterance.transcript]
        if start < min_start:
            min_start = start
        if end > max_end:
            max_end = end


    return X[:, min_start:max_end], y, dictionary


def get_X_y_waveform(utterances, max_length):
    dictionary = { word: i for i, word in enumerate(sorted(set(u.transcript for u in utterances))) }
    X = np.zeros((len(utterances), max_length, 1))
    y = np.zeros((len(utterances)), dtype=np.int)
    min_start = max_length
    max_end = 0
    for n, utterance in enumerate(utterances):
        fs, audio = data.audio.read_sound_file(utterance.filename)

        start = (X.shape[1] - audio.shape[0]) // 2
        end = start + audio.shape[0]
        X[n, start:end, 0] = audio
        y[n] = dictionary[utterance.transcript]
        if start < min_start:
            min_start = start
        if end > max_end:
            max_end = end


    return X[:, min_start:max_end], y, dictionary


