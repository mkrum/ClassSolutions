#! /usr/bin/env python2.7
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

def convert_onehot(val):
    base = np.zeros(3)
    base[val] = 1
    return base

def get_data():
    lines   = open('iris.csv', 'r').readlines()
    data    = []
    labels  = []
    for line in lines:
        spl = line.split(',')
        data.append([  float(index) for index in spl[:4] ])
        labels.append(spl[4].rstrip())

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    le = LabelEncoder()
    train_labels    = le.fit_transform(train_labels)
    test_labels     = le.transform(test_labels)
    train_labels    = [ convert_onehot(trlabel) for trlabel in train_labels ]
    test_labels     = [ convert_onehot(telabel) for telabel in test_labels ]

    return train_data, train_labels, test_data, test_labels

