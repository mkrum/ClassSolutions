#! /usr/bin/env python2.7

import theano as th
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from dataHandling import get_data

class Network:
    def __init__(self, dims):
        weights = []
        biases  = []
        for i in range(len(dims) - 1):
            weights.append(th.shared(create_weight(dims[i], dims[i + 1])))
            biases.append(th.shared(create_bias(dims[i + 1], 0.0, 0.1)))

        inputs  = T.vector()
        lr      = T.scalar()
        target  = T.vector()
        y       = inputs

        for w_vec, b_vec in zip(weights, biases):
            y = ff(y, w_vec, b_vec)

        cost = ((target - y)**2).mean(axis=0).sum()
        
        updates = []
        for w in weights:
            updates.append((w, w - lr*T.grad(cost, w)))

        for b in biases:
            updates.append((b, b - lr*T.grad(cost, b)))

        self.train_step = th.function([inputs, target, lr], cost,
                                       updates=updates)

        self.pred = th.function([inputs], y)

    def test_data(self, data, label):
        predlabel = self.pred(data)
        return np.argmax(predlabel) == np.argmax(label)

    def test_set(self, data, labels):
        results = []
        for d, l in zip(data, labels):
            results.append(self.test_data(d, l))
        return sum(results)/float(len(results))

if __name__ == '__main__':

    net         = Network([4, 5, 3])
    lr          = 0.01
    error       = []
    averages    = []

    train_data, train_labels, test_data, test_labels = get_data()
    for _ in range(1000):
        for i in xrange(len(train_data)):
            c = net.train_step(train_data[i], train_labels[i], lr)
            if i % 1000 == 0:
                error.append(c)
        averages.append(net.test_set(test_data, test_labels))

    plt.plot(averages)
    plt.show()
