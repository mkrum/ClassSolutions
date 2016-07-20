#! /usr/bin/env python2.7

import numpy as np
import tensorflow as tf
from math import sin, cos
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_hidden  =  128
n_rows    =  28
n_vals    =  28
n_outputs =  10

x = tf.placeholder(tf.float32, [None, n_rows, n_vals])
y = tf.placeholder(tf.float32, [None, n_outputs])

weights =  tf.Variable(tf.truncated_normal([n_hidden, n_outputs], stddev=0.1))
biases  =  tf.Variable(tf.random_normal([n_outputs]))

x_m =  tf.transpose(x, [1, 0, 2])
x_m =  tf.reshape(x_m, [-1, n_vals])
x_m =  tf.split(0, n_rows, x_m)

lstm_cell       =  tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
outputs, states =  tf.nn.rnn(lstm_cell, x_m, dtype=tf.float32)
pred            =  tf.matmul(outputs[-1], weights) + biases

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
      batch_x, batch_y = mnist.train.next_batch(50)
      batch_x = batch_x.reshape((50, 28, 28))
      optimizer.run(feed_dict={x: batch_x, y:batch_y})
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch_x, y:batch_y})

        print("step %d, training accuracy %g"%(i, train_accuracy))
    testimages = mnist.test.images
    testimages = testimages.reshape((len(mnist.test.labels), 28, 28))
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: testimages, y: mnist.test.labels}))
