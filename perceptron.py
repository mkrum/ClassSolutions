#! /usr/bin/env python2.7

import theano as th
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from utils import *

class Perceptron:
    def __init__(self, n_in):
        W = th.shared(create_weight(n_in, 1))
        b = th.shared(create_bias(1, 0, 0))

        inputs = T.vector() 
        target = T.scalar()
        lr = T.scalar()

        y = ff(inputs, W, b) 

        cost = ((target - y)**2).mean(axis=0).sum()

        gW = T.grad(cost, W)
        gb = T.grad(cost, b)
        
        self.train_step = th.function([inputs, target, lr], cost,
                                           updates=[(W, W - lr*gW),
                                                    (b, b - lr*gb)])

if __name__ == '__main__':
    ner = Perceptron(4)
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
