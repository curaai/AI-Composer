from mido import MidiFile
import mido
import numpy as np
from sklearn.utils import shuffle


class DataSet:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        
        self.min_ = 0 
        self.max_ = 0 

        self.notes = list()

    def generate_notes(self, midi_path):
        music = MidiFile(midi_path)
        
        for msg in music:
            if msg.time != 0:
                note = list()
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

    def get_feed_data(self):
        x_data = list()
        y_data = list()

        for i in range(len(self.notes) - self.seq_length - 1):
            x_data.append(self.notes[i: i + self.seq_length])
            y_data.append(self.notes[i + self.seq_length])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_data, y_data = shuffle(x_data, y_data, random_state=5)
        return x_data, y_data


if __name__ == '__main__':
    path = "test.mid"
    data = DataSet(15)
    data.generate_notes(path)
    data.pre_process_note()
    
    x, y = data.get_feed_data()

    print(x.shape)
    print(y.shape)