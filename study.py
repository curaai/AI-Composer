import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import midi_util


(seq, notes_dict) = midi_util.song2seq('test.mid')
seq_shape = seq.shape

seq_length = seq.shape[0]

hidden_size = seq_shape[1]
input_size = seq_shape[1]
num_classes = seq.shape[1]
learning_rate = 0.1

seq_length -= 1

batch_size = 1# ???


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

seq_loss = tf.contrib.seq2seq.sequence_loss (
    logtis=outputs, targets=Y, weights=weights
)
mean_loss = tf.reduce_mean(seq_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

