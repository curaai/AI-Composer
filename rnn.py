import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import midi_util


class Composer:
    def __init__(self):
        self.num_layers = 2
        self.hidden_size = 200
        self.dropout_prob = 0.8
        self.learning_rate = 5e-3
        self.num_epoch = 250

    def train(self):
        (seq, notes_dict) = midi_util.song2seq('test.mid')
        seq_shape = seq.shape

        seq_length = seq.shape[0]

        hidden_size = seq_shape[1]
        input_size = seq_shape[1]
        num_classes = seq.shape[1]
        learning_rate = 0.1

        seq_length -= 1

        batch_size = seq.shape[0] # ???

        X = tf.placeholder(tf.float32, [None, input_size])
        Y = tf.placeholder(tf.int32, [None, input_size])

        def lstm_cell():
            return rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

        multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(
            multi_cells,
            X,
            dtype=tf.float32)
        FC = tf.reshape(outputs, [-1, hidden_size])
        outputs = tf.contrib.layers.fully_connected(FC, num_classes, activation_fc=None)

        weights = tf.ones([batch_size, seq_length])

        seq_loss = tf.contrib.seq2seq.sequence_loss(
            logtis=outputs, targets=Y, weights=weights
        )
        mean_loss = tf.reduce_mean(seq_loss)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

        sess = tf.Session()
        sess.run(tf.global_varables_initializer())

        for i in range(self.num_epoch):
            _, l, results = sess.run([train_op, mean_loss, outputs],
                                     feed_dict={X: x_da})