from mido import MidiFile
import mido
import numpy as np


# shape ( mp3 length , 3 )
def song2note(file_name):
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

    note_dic = note_dict(notes)

    return np.array(notes), note_dic


def encoding(notes, time_line, note_dic):
    note_seq = []
    note_len = []
    i = 0
    while True:
        try:
            j = i
            line = [note_dic[notes[j]]]
            while time_line[j] == time_line[j + 1]:
                j += 1
                line.append(note_dic[notes[j]])
            i = j + 1

            note_len.append(len(line))
            note_seq.append(line)
        except IndexError:
            break

    return np.array(note_seq), np.array(note_len)


def note_dict(note):
    return {pitch: label for label, pitch in enumerate(list(set(note)))}


def make_xy(sentence, seq_length):
    x_data = []
    y_data = []
    for i in range(0, len(sentence) - seq_length):
        x_seq = sentence[i: i + seq_length]
        y_seq = sentence[i+1: i + seq_length + 1]

        x_data.append(x_seq)
        y_data.append(y_seq)

    return np.array(x_data), np.array(y_data)
