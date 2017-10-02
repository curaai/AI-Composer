from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import midi_util

window_size = 10

notes, note_dic = midi_util.song2seq('test.mid')
x, y = midi_util.seq2data(notes, window_size)
shape = y.shape

n_class = len(note_dic)
n_epoch = 250
drop_out = 0.7
batch_size = 1


model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(batch_size, shape, window_size)))
model.add(Dropout(0.7))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.7))
model.add(Dense(n_class))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(x, y, batch_size=1, nb_epoch=n_epoch)
