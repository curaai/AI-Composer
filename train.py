import rnn
import tensorflow as tf
import midi_util
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    path = 'test.mid'

    with tf.Session() as sess:
        c = rnn.Composer(sess, path)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./board/composer', graph=sess.graph)
        print('Learning Start!')

        for i in range(c.num_epoch):
            x_data, y_data = midi_util.make_xy(c.seq, c.seq_length)
            _, l, summary = c.train(x_data, y_data)

            if i % 5 == 0:
                writer.add_summary(summary, float(i))
