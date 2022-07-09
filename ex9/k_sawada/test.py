import pickle
import os
import datetime

from tensorflow.python.keras import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import librosa
from keras.utils import np_utils

SAMPLING_RATE = 8000  # by wave format data
LABELS = 10  # number of labels to classify
DATA_LENGTH = 4096


def load_wav_test():
    """load test wave file

    Returns:
        ndarray, axis=(sample, time): wav
        ndarray, axis=(sample): answer label
    """
    if os.path.isfile('wav_test.pickle') and os.path.isfile('ans_test.pickle'):
        with open('wav_test.pickle', 'rb') as f:
            wav = pickle.load(f)
        with open('ans_test.pickle', 'rb') as f:
            label = pickle.load(f)
    else:
        test_csv = pd.read_csv("./test_truth.csv", dtype=str, encoding='utf8')
        wav = np.zeros((len(test_csv), DATA_LENGTH))
        label = np.zeros(len(test_csv), dtype=np.int16)
        for i, row in test_csv.iterrows():
            wav_tmp, _ = librosa.load(f"../{row.path}", sr=SAMPLING_RATE, mono=True)
            # zero padding at end of wav
            if (len(wav_tmp) > DATA_LENGTH):
                wav[i] = wav_tmp[0:DATA_LENGTH]
            else:
                wav[i, 0:len(wav_tmp)] = wav_tmp
            label[i] = row.label
        with open('wav_test.pickle', 'wb') as f:
            pickle.dump(wav, f)
        with open('ans_test.pickle', "wb") as f:
            pickle.dump(label, f)
    return wav, label


def main():
    # read data
    x, answer_label = load_wav_test()

    # convert to mfcc
    tmp_test = []
    for i in range(len(x)):
        tmp_test.append(librosa.feature.mfcc(x[i]))
    x = np.array(tmp_test)

    # load model
    model_arc_filename="./trained_model/2022-07-05 23:25:48.287827model_architecture.json"
    model_weight_filename="./trained_model/2022-07-05 23:25:48.287827model_weight.hdf5"
    model_arc_str = open(model_arc_filename).read()
    model = models.model_from_json(model_arc_str)
    model.load_weights(model_weight_filename)

    # predict
    predict = model.predict(x, verbose=0)
    predict = np.argmax(predict, axis=1)  # decode one-hot

    # accuracy
    collect_count = np.count_nonzero(answer_label == predict)
    all_count = len(answer_label)
    accuracy = accuracy_score(answer_label, predict)
    print(f"accuracy: {(accuracy)}")
    print(f"        ( {collect_count} / {all_count} )")

    # plot result
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    labels = [i for i in range(LABELS)]
    column_labels = [f"pred_{i}" for i in labels]
    row_labels = [f"ans_{i}" for i in labels]

    ax.set_title(
        "test\n" +
        f"acc:{accuracy:.6f}")
    cm = confusion_matrix(answer_label, predict,
                                  labels=labels)
    ax.pcolor(cm, cmap=plt.cm.Blues)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.5, '{}'.format(z),
                   ha='center', va='center', color="orange")

    ax.set_xticks(np.arange(cm.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(cm.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    plt.savefig(f"result{datetime.datetime.now()}.png")


if __name__ == "__main__":
    main()
