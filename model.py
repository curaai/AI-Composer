import tensorflow as tf
from tensorflow.contrib import rnn


class Composer:
    def __init__(self, sess, seq_length):
        self.sess = sess

        self.learning_rate = 0.01

        self.seq_length = seq_length
        self.hidden_size = 128
        self.rnn_size = 4
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
        self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope("rnn"):
            multi_cells = rnn.MultiRNNCell([_lstm_cell() for _ in range(self.rnn_size)], state_is_tuple=True)
            output, states_ = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)

            pred = tf.contrib.layers.fully_connected(output[:, -1], self.output_size, activation_fn=None)
            self.pred = pred

            # mse loss
            self.cost = tf.losses.mean_squared_error(self.Y, self.pred)
            # adam optimizer
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            self.optimizer = optimizer

            correct_pred = tf.equal(self.pred, self.Y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
            self.saver = tf.train.Saver()

            self.sum_cost = tf.summary.scalar('cost', self.cost)
            self.sum_accuracy = tf.summary.scalar('accuracy', self.accuracy)
            self.summary = tf.summary.merge_all()

    def train(self, x, y, is_training=True, keep_prob=0.2):
        return self.sess.run([self.optimizer, self.pred, self.accuracy, self.cost, self.summary],
                             feed_dict={self.X: x, self.Y: y, self.keep_prob: keep_prob, self.is_training: is_training})

    def predict(self, x, is_training=False, keep_prob=1.0):
        return self.sess.run([self.pred], feed_dict={self.X: x, self.keep_prob: keep_prob, self.is_training: is_training})
