from mido import MidiFile
import mido
import numpy as np


# shape ( mp3 length , 3 )
def song2sequence(file_name):
    mid = MidiFile(file_name)
    is_note = lambda x: isinstance(x, mido.Message) and x.type in ["note_on", "note_off"]

    current_tick = 0
    notes = list()
    time_line = list()

    for track in mid.tracks:
        for msg in track:
            current_tick += msg.time
            if is_note(msg):
                time_line.append(current_tick)
                notes.append(msg.note)

    notes_dict = note_dict(notes)
    dict_len = len(notes_dict)

    sequence = list()
    i = 0
    while True:
        try:
            j = i

            line = np.zeros(dict_len)
            line[notes_dict[notes[j]]] = 1

            while time_line[j] == time_line[j+1]:
                j += 1
                line[notes_dict[notes[j]]] = 1
            i = j + 1

            sequence.append(line)
        except IndexError:
            break

    return sequence


def note_dict(note):
    return {pitch: label for label, pitch in enumerate(list(set(note)))}


if __name__ == '__main__':
    for line in song2sequence('test.mid'):
        print(line)
