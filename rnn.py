import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import midi_util


class Composer:
    def __init__(self, sess, file_path, window_size):
        seq, notes_dict = midi_util.song2seq(file_path, window_size)
        self.seq = seq
        self.notes_dict = notes_dict

        self.num_layers = 2
        self.learning_rate = 5e-3
        self.num_epoch = 250
        self.notes_dict = notes_dict
        note_class = len(notes_dict)

        self.n_input = note_class
        self.n_batch = len(seq) - window_size
        self.n_hidden = 200
        self.n_class = note_class
        self.seq_length = window_size

        self.sess = sess
        self.build()

    def build(self):
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.n_class])
        self.Y = tf.placeholder(tf.float32, [None, self.seq_length])
        self.W = tf.Variable(tf.random_normal([self.n_hidden, self.n_class]))
        self.b = tf.Variable(tf.random_normal([self.n_class]))

        multi_cells = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)], state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(
            multi_cells,
            self.X,
            dtype=tf.float32)

        # output [batch_size, seq_length, n_hidden]
        x_for_softmax = tf.reshape(outputs, [-1, self.n_hidden])

        softmax_w = tf.get_variable("softmax_w", [self.n_hidden, self.n_class])
        softmax_b = tf.get_variable("softmax_b", [self.n_class])
        outputs = tf.matmul(x_for_softmax, softmax_w) + softmax_b

        outputs = tf.reshape(outputs, [self.n_batch, self.seq_length, self.n_class])
        weights = tf.ones([self.n_batch, self.seq_length, self.n_class])

        self.Y = tf.reshape(self.Y, [-1])
        seq_loss = tf.contrib.seq2seq.sequence_loss(
            logits=outputs, targets=self.Y, weights=weights
        )

        self.saver = tf.train.Saver()

        self.mean_loss = tf.reduce_mean(seq_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        self.pred = tf.nn.softmax(outputs)

    def train(self, x_data, y_data):
        return self.sess.run([self.train_op, self.mean_loss],
                             feed_dict={self.X: x_data, self.Y: y_data})

    def lstm_cell(self):
        return rnn.BasicLSTMCell(self.n_hidden, state_is_tuple=True)

    def predict(self, X):
        return self.sess.run(self.pred, feed_dict={self.X: X})
