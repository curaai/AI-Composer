import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class Composer:
    def __init__(self, sess, seq_length):
        self.sess = sess

        self.learning_rate = 0.01

        self.seq_length = seq_length
        self.hidden_size = 128
        self.rnn_size = 2
        self.output_size = 3

        self.build()

    def build(self):
        def _lstm_cell():
            cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell

        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.output_size])
        self.Y = tf.placeholder(tf.float32, [None, self.output_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name='phase')

        multi_cells = rnn.MultiRNNCell([_lstm_cell() for _ in range(self.rnn_size)], state_is_tuple=True)
        output, states_ = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)

        with tf.variable_scope("rnn"):
            with tf.name_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(output, 1024)
                fc1 = tf.contrib.layers.batch_norm(fc1,
                                                   center=True, scale=True,
                                                   is_training=self.is_training)
                fc1 = tf.nn.relu(fc1, name='relu1')

            with tf.name_scope("fc2"):
                fc2 = tf.contrib.layers.fully_connected(fc1, 1024)
                fc2 = tf.contrib.layers.batch_norm(fc2,
                                                   center=True, scale=True,
                                                   is_training=self.is_training)
                fc2 = tf.nn.relu(fc2, name='relu1')

            pred = tf.contrib.layers.fully_connected(fc2[:, -1], self.output_size)
            self.pred = pred

            self.cost = tf.reduce_sum(tf.pow(self.pred - self.Y, 2))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.optimizer = optimizer

            correct_pred = tf.equal(self.pred, self.Y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self, x, y, is_training=True, keep_prob=0.7):
        return self.sess.run([self.optimizer, self.pred, self.accuracy],
                             feed_dict={self.X: x, self.Y: y, self.keep_prob: keep_prob, self.is_training: is_training})

    def predict(self, x, is_training=False, keep_prob=1.0):
        return self.sess.run([self.pred], feed_dict={self.X: x, self.keep_prob: keep_prob, self.is_training: is_training})
