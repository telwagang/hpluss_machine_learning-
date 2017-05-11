import csv
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pickle

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def train(value):

    model = None
    model_path = ROOT_DIR + '/Datasets/Models/'
    dataset_path = ROOT_DIR + '/Datasets/files/training.csv'

    if value == 'train':
        x_train, y_train = read_csv(dataset_path)
        x, y = scale(x_train), y_train

        model = LinearRegression(normalize=True)
        model.fit(x, y)
        pickle.dump(model, open(model_path + 'model.sav', 'wb'))

    elif value == 'load':
        model = pickle.load(open(model_path + 'model.sav', 'rb'))

    return model


def predict(model, test):
    _test = np.array(test)
    __test = _test.reshape(1, -1)
    pre = model.predict(__test)
    return pre


def read_csv(filename):
    dataset = []
    labels = []
    with open(filename, 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            feat = [float(i) for i in line[:-1]]
            label = float(line[-1])
            dataset.append(feat)
            labels.append(label)
    return dataset, labels


def write_csv(data):
    return
