from mido import MidiFile
import mido
import numpy as np
from sklearn.utils import shuffle


class DataSet:
    def __init__(self, midi_path, seq_length, batch_size):
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.min_ = 0 
        self.max_ = 0 
        self.notes = list()

        self.midi_path = midi_path

    def generate_notes(self):
        music = MidiFile(self.midi_path)
        
        for msg in music:
            if msg.time != 0:
                # note = note, velocity, time                
                if not msg.is_meta and msg.type == 'note_on':
                    note = msg.bytes()[1:] + [msg.time]
                else:
                    note = [0, 0, msg.time]
                self.notes.append(note)

    def pre_process_note(self):
        notes = np.array(self.notes)

        note = notes[:, 0]
        velocity = notes[:, 1]
        time = notes[:, 2]

        note     /= max(note)
        velocity /= max(velocity)
        time     /= max(time)

        # (len) * 3  =>  (len, 3)
        notes = np.dstack((note, velocity, time))[0]
        self.notes = notes

        x_data = list()
        y_data = list()

        for i in range(len(self.notes) - self.seq_length - 1):
            x_data.append(self.notes[i: i + self.seq_length])
            y_data.append(self.notes[i + self.seq_length])

        self.x = np.array(x_data)
        self.y = np.array(y_data)

    def get_feed_data(self):
        total_size = len(self.y)
        mask = np.random.choice(total_size, self.batch_size)

        return self.x[mask], self.y[mask]
