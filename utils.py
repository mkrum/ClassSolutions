#! /usr/bin/env python2.7

import theano as th
import theano.tensor as T
import numpy as np

dtype = th.config.floatX

def create_weight(n_in, n_out):
    vals = np.ndarray([n_in, n_out], dtype=dtype)
    for w in range(n_in):
        row = np.random.uniform(low=-1/np.sqrt(n_in), high=1/np.sqrt(n_in), size=(n_out,))
        vals[w,:] = row
    return vals

def create_bias(n_out, low, high):
    return np.cast[dtype](np.random.uniform(low, high, size=n_out))

def ff(inputs, weights, bias):
    return T.nnet.sigmoid(T.dot(inputs, weights) + bias)
