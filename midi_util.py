from mido import MidiFile
import mido
import numpy as np


def extract_mido(mido_path):
    time = 0
    prev = 0
    notes = list()

    music = MidiFile(mido_path)
    for msg in music:
        time += msg.time

        if not msg.is_meta and msg.type == 'note_on':
            # note, velocity, time
            note = msg.bytes()[1:] + [time-prev]
            prev = time
            notes.append(note)   

    return notes


def pre_process_note(notes):
    notes = np.array(notes)
    note = notes[:, 0]
    velocity = notes[:, 1]
    time = notes[:, 2]

    print(max(note))
    print(max(velocity))
    print(max(time))

if __name__ == '__main__':
    path = "test.mid"
    notes = extract_mido(path)
    pre_process_note(notes)