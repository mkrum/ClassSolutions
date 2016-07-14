#! /usr/bin/env python2.7

import theano as th
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from utils import *

class Network:
    def __init__(self, dims):
        weights = []
        biases  = []
        for i in range(len(dims) - 1):
            weights.append(th.shared(create_weight(dims[i], dims[i + 1])))
            biases.append(th.shared(create_bias(dims[i + 1], 0.0, 0.1)))

        inputs = T.vector()
        lr = T.scalar()
        target = T.scalar()
        y = inputs

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

if __name__ == '__main__':
    ner = Network([4, 5,1])
    lr = 0.01
    vals = []
    for i in xrange(int(100000)):
        randInputs = np.random.rand(4)
        # random function to learn
        target = randInputs[0]*randInputs[3]
        c = ner.train_step(randInputs, target, lr)
        if i % 1000 == 0:
            vals.append(c)

    plt.plot(vals)
    plt.show()
