import tensorflow as tf
import argparse
import numpy as np
import os
from mido import Message, MidiFile, MidiTrack

import model
import midi_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_song(notes, max_time, path):
    print("generte songs") 
    print(path)

    new_notes = list()
    for _note in notes:
        tone = _note[0][0]
        velocity =_note[0][1]
        time = _note[0][2]

        tone = int(tone * 88 + 24)
        velocity = int(velocity * 127)
        time = int(time * max_time/0.001025)

        if tone < 24:
            tone = 24
        elif tone > 102:
            tone = 102

        if velocity < 0:
            velocity = 0
        elif velocity > 127:
            velocity = 127

        if time < 0:
            time = 0

        new_notes.append([tone, velocity, time])

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note in new_notes:
        track.append(Message('note_on', note=note[0], velocity=note[1], time=note[2]))
        print('note: {}, velocity: {}, time= {}'.format(note[0], note[1], note[2]))

    mid.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--seq_length', type=int, default=30, help='How many set sequence length to train')
    parser.add_argument('-nl', '--note_length', type=int, default=3000, help='Note length of music')
    parser.add_argument('-s', '--save', default='save/save.ckpt', help='Save trained network')
    parser.add_argument('-m', '--midi', default='test.mid', help="Head of seed midi notes")
    parser.add_argument('-p', '--path', help="Save composed midi song", required=True)
    args = parser.parse_args()

    data = midi_util.DataSet(args.seq_length, 1)
    data.generate_notes(args.midi)
    data.pre_process_note()

    # initial data to generate notes
    sequence = data.x[0].reshape(1, 30, 3)
    pred_notes = list()

    print(args.path)
    with tf.Session() as sess:
        composer = model.Composer(sess, args.seq_length)
        sess.run(tf.global_variables_initializer())
        composer.saver.restore(sess, args.save)

        for i in range(args.note_length):
            # before notes [1:] + predicted note []
            note = composer.predict(sequence)[0]
            sequence = np.vstack((sequence[0][1:], note))
            # stack generated note
            sequence = sequence.reshape(1, 30, 3)
            pred_notes.append(note)

    time_path = os.path.join(os.path.dirname(args.save), 'maxtime.txt')
    with open(time_path, 'r') as f:
        time = float(f.readline())

    print(time)
    
    generate_song(pred_notes, time, args.path)

