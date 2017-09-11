import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import midi_util


class Composer:
    def __init__(self, sess, file_path):
        (seq, notes_dict) = midi_util.song2note(file_path)
        self.seq = seq
        self.notes_dict = notes_dict

        self.num_layers = 2
        self.learning_rate = 5e-3
        self.num_epoch = 250
        self.notes_dict = notes_dict
        note_class = len(notes_dict)

        self.batch_size = seq.shape[0]
        self.hidden_size = 200
        self.num_classes = note_class
        self.seq_length = 15

        self.sess = sess
        self.build()

    def build(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_length])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_length])
        self.batch_size = tf.shape(self.X)[0]

        X_one_hot = tf.one_hot(self.X, self.num_classes)

        multi_cells = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)], state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(
            multi_cells,
            X_one_hot,
            dtype=tf.float32)

        X_for_fc = tf.reshape(outputs, [-1, self.hidden_size])
        outputs = tf.contrib.layers.fully_connected(X_for_fc, self.num_classes, activation_fn=None)

        outputs = tf.reshape(outputs, [self.batch_size, self.seq_length, self.num_classes])

        weights = tf.ones([self.batch_size, self.seq_length])

        seq_loss = tf.contrib.seq2seq.sequence_loss(
            logits=outputs, targets=self.Y, weights=weights
        )

        self.saver = tf.train.Saver()

        self.mean_loss = tf.reduce_mean(seq_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        self.pred = tf.nn.softmax(outputs)

        self.summary = tf.summary.merge_all()

    def train(self, x_data, y_data):
        return self.sess.run([self.train_op, self.mean_loss, self.summary],
                             feed_dict={self.X: x_data, self.Y: y_data})

    def lstm_cell(self):
        return rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)

    def predict(self, X):
        return self.sess.run(self.pred, feed_dict={self.X: X})
