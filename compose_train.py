import argparse
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import compose_model
import midi_util


if __name__ == '__main__':
    exmaple_text = ''' example:

    python compose_train.py -sl 20 -b 30 -i 1000 -s save/save.ckpt -m a.midi,b.midi,c.midi'''

    parser = argparse.ArgumentParser(epilog=exmaple_text)
    parser.add_argument('-sl', '--seq_length', type=int, default=30, help='How many set sequence length to train')
    parser.add_argument('-b', '--batch_size', type=int, default=30, help="Batch size")
    parser.add_argument('-i', '--iteration', type=int, default=2000, help='Iteration')
    parser.add_argument('-s', '--save', default='save/save.ckpt', help='Save trained network')
    parser.add_argument('-m', '--midi', default='test.mid', help="File to train")
    args = parser.parse_args()

    data = midi_util.DataSet(args.seq_length, args.batch_size)
    midi = args.midi.split(",")
    for path in midi:
        data.generate_notes(path)
    data.pre_process_note()

    with tf.Session() as sess:
        composer = compose_model.Composer(sess, args.seq_length)
        sess.run(tf.global_variables_initializer())
        print("Learning Start !!!")

        for i in range(args.iteration):
            x, y = data.get_feed_data()
            _, pred, accuracy = composer.train(x, y)
            if i % 10 == 0:
                print("Iteration: {0}, Accuracy: {1}".format(i, accuracy))

        composer.saver.save(sess, args.save)

        path = os.path.join(os.path.dirname(args.save), 'maxtime.txt')
        with open(path, 'w') as f:
            f.write(str(data.max_time))
        print("Learning Finish !!!")   


