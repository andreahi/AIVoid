import glob
import os
import pickle

import time

import re

from FFModel import FFModel
from Test import runTest

max_data_size = 1000000

data = []
labels = []
model = FFModel(11)

while 1:
    labels_files = []
    data_files = []
    while len(labels_files) < 1:

        files_path = os.path.join("trainingdata", 'labels*')
        labels_files = sorted(
            glob.iglob(files_path), key=os.path.getctime, reverse=True)
        #hash = re.findall(r'labels_(.*)\.pickle', labels_files[0])[0]

        files_path = os.path.join("trainingdata", 'data*')
        data_files = sorted(
            glob.iglob(files_path), key=os.path.getctime, reverse=True)
        if len(labels_files) < 1:
            print "waiting"
            time.sleep(10)


    for f in labels_files[:1]:
        with open(f, 'rb') as handle:
            labels = labels + pickle.load(handle)
            os.remove(f)

    for f in data_files[:1]:
        with open(f, 'rb') as handle:
            data = data + pickle.load(handle)
            os.remove(f)


    print "traning new model"
    model.train(data, labels)
    model.save()
    #runTest()

    data = []
    labels = []
    if len(labels) > max_data_size:
        del data[: len(data) - max_data_size]
        del labels[: len(labels) - max_data_size]


    #if len(labels_files) >= 10:
    #    print "deleting ", labels_files[-1]
    #    print "deleting ", data_files[-1]
    #    os.remove(labels_files[-1])
    #    os.remove(data_files[-1])
        # del labels_files[-1]
        # del data_files[-1]