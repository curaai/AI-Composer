import tensorflow as tf 
import numpy as np 
from tensorflow.contrib import rnn 


class Model:
    def __init__(self, sess, seq_legnth, batch_size):
        self.sess = sess

        self.learning_rate = 0.001

        self.hidden_size = 128
        self.rnn_size    = 2
        self.output_size = 3

    def build(self):
        def _lstm_cell():
            cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            return cell

        self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.output_size])
        self.Y = tf.placeholder(tf.float32, [None, self.output_size])

        multi_cells = rnn.MultiRNNCell([_lstm_cell() for _ in range(self.rnn_size)], state_is_tuple=True)
        output, states_ = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)

        with tf.name_scope("fully connected 1"):
            weight1 = tf.get_variable("weight1", [self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
            bias1 = tf.get_variable("bias1", [self.hidden_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
            
            fc1 = tf.matmul(ouput, weight1) + bias1

        with tf.name_scope("fully connected 2"):
            weight2 = tf.get_variable("weight2", [self.hidden_size, self.output_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
            bias2 = tf.get_variable("bias2", [self.hidden_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
            
            pred = tf.matmul(fc1, weight2) + bias2

        cost = tf.reduce_mean(tf.square(pred - y, 2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        correct_pred = tf.equal(pred, y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
