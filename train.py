import tensorflow as tf

import model
import midi_util

SEQ_LENGTH = 30
BATCH_SIZE = 300
ITERATION = 10000

if __name__ == '__main__':
    midi = "test.mid"
    data =  midi_util.DataSet(midi, SEQ_LENGTH, BATCH_SIZE)
    data.generate_notes()
    data.pre_process_note()

    with tf.Session() as sess:
        composer = model.Composer(sess, SEQ_LENGTH)
        sess.run(tf.global_variables_initializer())
        print("Learning Start !!!")

        for i in range(ITERATION):
            x, y = data.get_feed_data()
            _, pred, accuracy = composer.train(x, y)
            if i % 10 == 0:
                print("Iteration: {0}, Accuracy: {1}".format(i, accuracy))

        print("Learning Finish !!!")        
