from mido import MidiFile
import mido


def test(file_name):
    mid = MidiFile(file_name)
    is_note = lambda msg: isinstance(msg, mido.Message) and msg.type in ["note_on", "note_off"]
    music = [[msg.note, msg.time, msg.velocity] for track in mid.tracks for msg in track if is_note(msg)]

    return music
