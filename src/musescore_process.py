from miditoolkit.midi import parser as mid_parser
from vocab import Vocab
from data_process import melody_extraction, key_sig_recognition
import os
import argparse
import uuid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
    )
    args = parser.parse_args()
    midi_obj = mid_parser.MidiFile(args.file)
    # midi_obj.instruments[0].notes += midi_obj.instruments[1].notes
    # del midi_obj.instruments[1]
    # del midi_obj.instruments[0]
    # temp = midi_obj.instruments[0]
    # midi_obj.instruments[0] = midi_obj.instruments[1]
    # midi_obj.instruments[1] = temp

    min_step = midi_obj.ticks_per_beat * 4 / 16
    for i in range(len(midi_obj.instruments)):
        for n in range(len(midi_obj.instruments[i].notes)):
            midi_obj.instruments[i].notes[n].start = int(
                int(midi_obj.instruments[i].notes[n].start / min_step) *
                min_step)
            midi_obj.instruments[i].notes[n].end = int(
                int(midi_obj.instruments[i].notes[n].end / min_step) * min_step)

    melody_notes = midi_obj.instruments[0].notes

    melody_notes.sort(key=lambda x: (x.start, -x.pitch))

    bins = []
    prev = None
    tmp_list = []
    for nidx in range(len(melody_notes)):
        note = melody_notes[nidx]

        if note.start != prev:
            if tmp_list:
                bins.append(tmp_list)
            tmp_list = [note]
        else:
            tmp_list.append(note)
        prev = note.start
    bins.append(tmp_list)

    # preserve only highest one at each step
    notes_out = []
    for b in bins:
        notes_out.append(b[0])

    # avoid overlapping
    notes_out.sort(key=lambda x: x.start)
    for idx in range(len(notes_out) - 1):
        if notes_out[idx].end >= notes_out[idx + 1].start:
            notes_out[idx].end = notes_out[idx + 1].start

    # delete note having no duration
    notes_clean = []
    for note in notes_out:
        if note.start != note.end:
            notes_clean.append(note)

    # filtered by interval
    notes_final = [notes_clean[0]]

    for i in range(1, len(notes_clean) - 1):
        if ((notes_clean[i].pitch - notes_clean[i-1].pitch) <= -9) and \
        ((notes_clean[i].pitch - notes_clean[i+1].pitch) <= -9):
            continue
        else:
            notes_final.append(notes_clean[i])

    notes_final += [notes_clean[-1]]
    midi_obj.instruments[0].notes = notes_final

    output_file = os.path.join("train_mid/musescore/mid_1", args.file)
    midi_obj.dump(output_file)
    myvocab = Vocab()
    remi_seq = myvocab.midi2REMI(output_file, dataset_name="wikifonia")
    myvocab.REMIID2midi(remi_seq, output_file)
    melody = melody_extraction(output_file)
    key = key_sig_recognition(myvocab, remi_seq)

    nounce = str(uuid.uuid4())
    melody_file = f"/tmp/{nounce}_melody.txt"
    with open(melody_file, "w") as f:
        for m in melody:
            f.write("{} {}\n".format(*m))
    f.close()

    key_file = f"/tmp/{nounce}_key.txt"
    with open(key_file, "w") as f:
        f.write(key)
    f.close()

    file = args.file.replace(".mid", ".txt")
    phrase_label_file = os.path.join("train_mid/musescore/phrase_label", file)
    open(phrase_label_file, "w").close()
    import subprocess
    phrase_result = subprocess.check_output(["./seg", melody_file, key_file],
                                            timeout=60 * 5).decode().strip()

    print(phrase_result)
    with open(phrase_label_file, "w") as f:
        f.write(phrase_result)
