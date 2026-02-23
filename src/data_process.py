import os
from vocab import Vocab
import pickle
from tqdm import tqdm
import parse
import argparse
from mido import MidiFile
import mido
import uuid
from key_finding import key_sig_recognition
import subprocess


def raw_melody(file, print_details=False):

    mid = MidiFile(file)

    abs_time = 0
    mldy_events_seconds = []
    mldy_events_ticks = []
    mldy_events = []
    for msg in mid:
        # time.sleep(msg.time)
        abs_time += msg.time

        if (not msg.is_meta) and msg.channel == 0:
            msg.time = abs_time
            mldy_events_seconds.append(msg)
            # print(msg)

    for msg in mid.tracks[1]:
        if not msg.is_meta:
            mldy_events_ticks.append(msg)
            # print(msg)

    for idx, msg in enumerate(mldy_events_ticks):
        if msg.type == "note_on":
            mldy_e = {
                "note": msg.note,
                "time_start": mldy_events_seconds[idx].time,
                "time_end": mldy_events_seconds[idx + 1].time,
            }
            mldy_events.append(mldy_e)
        # print(msg)
        # print(mldy_events_seconds[idx])
    if print_details:
        for e in mldy_events:
            print(e)
    return mldy_events


'''
melody_extraction
input: piece is three digit index as string
output: the melody in the form of:
	[MIDI_num, number of sixteenth notes] or
	[0, number of sixteenth notes]
The starting position is the same as the finalized_chord analysis, 
with an offset of begin_time, that's in the begin_time.txt
'''


def melody_extraction(file):
    # starting time in second

    begin_time = 0

    raw_mldy = raw_melody(file)

    midiFile = mido.MidiFile(file, clip=True)
    tempo = midiFile.tracks[0][0].tempo
    beat_length = mido.tick2second(120, 480, tempo)

    melody = []
    melody.append([
        0, 0,
        int(round(
            (float(raw_mldy[0]["time_start"]) - begin_time) / beat_length))
    ])

    # the same quantize method as the chord, given the starting and ending time
    # of the note in seconds
    for i in range(len(raw_mldy)):
        note = []
        note.append(raw_mldy[i]["note"])
        note.append(
            int(
                round((float(raw_mldy[i]["time_start"]) - begin_time) /
                      beat_length)))
        note.append(
            int(
                round((float(raw_mldy[i]["time_end"]) - begin_time) /
                      beat_length)))

        # some note duration is especially short (like staccato), to prevent it
        # having duration 0, we add to the time_end. So this means that for any
        # note it at least has duration of a sixteenth
        if note[-1] == note[-2]:
            note[-1] += 1
        melody.append(note)

    # for m in melody:
    # 	print(m)

    # fill the middle with rests
    tmp = []
    for idx, m in enumerate(melody):
        dur = m[2] - m[1]
        if idx > 0 and m[1] > melody[idx - 1][2]:
            tmp.append([0, m[1] - melody[idx - 1][2]])
        tmp.append([m[0], dur])

    melody = tmp

    return melody


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nottingham_dir", type=str)
    parser.add_argument("--wkifonia_dir", type=str)
    parser.add_argument("--theorytab_dir", type=str)
    parser.add_argument("--pop909_dir", type=str)

    args = parser.parse_args()

    myvocab = Vocab()

    remi_seqs = []
    midis = [
        f for f in os.listdir("train_mid/musescore/mid_1") if f.endswith(".mid")
    ]
    for midi in tqdm(midis,):
        remi_seq = myvocab.midi2REMI(os.path.join("train_mid/musescore/mid_1",
                                                  midi),
                                     include_bridge=False,
                                     verbose=False,
                                     dataset_name="musescore")
        myvocab.REMIID2midi(remi_seq, f"train_mid/musescore/mid_1/{midi}")

        phrase_file = f"train_mid/musescore/phrase_label/{midi}"
        phrase_file = phrase_file.replace(".mid", ".txt")
        if not os.path.exists(phrase_file):
            continue

        try:
            f = open(phrase_file, "r")
            text = f.read()
            f.close()
            assert text != ""
            phrase_labels, phrease_bar_nums = myvocab.parse_struct(phrase_file)
            assert all(x in myvocab.phrase_labels for x in phrase_labels)
            assert all(x.isupper() for x in phrase_labels)
            assert all(int(x) <= 35 for x in phrease_bar_nums)
        except:
            print(f"except for {midi}")
            continue

        try:
            remi_seq = myvocab.insert_phrase_label(remi_seq, phrase_file)
        except:
            remi_seq, bar_diff_num = myvocab.fix_phrase_label(
                remi_seq, phrase_file)
            if abs(bar_diff_num) >= 5:
                continue

        remi_seq = myvocab.insert_section_start_end(remi_seq)

        try:
            assert remi_seq.count(myvocab.token2id["Section_Start"]) == 1
            assert remi_seq.count(myvocab.token2id["Section_End"]) == 1
            assert myvocab.id2token[remi_seq[0]] == "Section_Start"
            assert myvocab.id2token[remi_seq[-1]] == "Section_End"
        except:
            continue

        myvocab.REMIID2midi(remi_seq, f"train_mid/musescore/mid_2/{midi}")

        remi_seqs.append(remi_seq)

    pickle.dump(remi_seqs,
                open(f"data_pkl/musescore.pkl", 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    datasets = [
        {
            "name": "theorytab",
            "path": args.theorytab_dir
        },
        {
            "name": "wikifonia",
            "path": args.wkifonia_dir
        },
        {
            "name": "nottingham",
            "path": args.nottingham_dir
        },
    ]

    error_midi_count = 0
    for dataset in datasets:
        midis = [f for f in os.listdir(dataset["path"]) if f.endswith(".mid")]
        remi_seqs = []

        for midi in tqdm(midis, desc=dataset["name"]):
            try:
                remi_seq = myvocab.midi2REMI(os.path.join(
                    dataset["path"], midi),
                                             include_bridge=False,
                                             verbose=False,
                                             dataset_name=dataset["name"])
            except:
                continue

            myvocab.REMIID2midi(remi_seq,
                                f"train_mid/{dataset['name']}/mid/{midi}")

            phrase_file = f"train_mid/{dataset['name']}/phrase_label/{midi}"
            phrase_file = phrase_file.replace(".mid", ".txt")

            if not os.path.exists(phrase_file):
                try:
                    melody = melody_extraction(
                        f"train_mid/{dataset['name']}/mid/{midi}")
                    key = key_sig_recognition(myvocab, remi_seq)
                except:
                    continue

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

                try:
                    phrase_result = subprocess.check_output(
                        ["./seg", melody_file, key_file],
                        timeout=60 * 2).decode().strip()
                except subprocess.TimeoutExpired:
                    print(f"timeout for {midi}")
                    phrase_result = ""

                phrase_result = phrase_result.replace("\n", "")
                with open(phrase_file, "w") as f:
                    f.write(phrase_result)
                f.close()

                os.remove(melody_file)
                os.remove(key_file)

            try:
                f = open(phrase_file, "r")
                text = f.read()
                f.close()
                assert text != ""
                phrase_labels, phrease_bar_nums = myvocab.parse_struct(
                    phrase_file)
                assert all(x in myvocab.phrase_labels for x in phrase_labels)
                assert all(x.isupper() for x in phrase_labels)
                assert all(int(x) <= 32 for x in phrease_bar_nums)
            except:
                print(f"except for {midi}")
                continue

            try:
                remi_seq = myvocab.insert_phrase_label(remi_seq, phrase_file)
            except:
                remi_seq, bar_diff_num = myvocab.fix_phrase_label(
                    remi_seq, phrase_file)
                with open('text_dir/error_phrase.txt', 'a') as file:
                    file.write(f"midi: {midi}, bar_diff_num: {bar_diff_num}\n")
                if abs(bar_diff_num) >= 5:
                    error_midi_count += 1
                    continue

            try:
                assert remi_seq.count(myvocab.token2id["Bar"]) == 0
            except:
                continue

            remi_seq = myvocab.insert_section_start_end(remi_seq)

            try:
                assert remi_seq.count(myvocab.token2id["Section_Start"]) == 1
                assert remi_seq.count(myvocab.token2id["Section_End"]) == 1
                assert myvocab.id2token[remi_seq[0]] == "Section_Start"
                assert myvocab.id2token[remi_seq[-1]] == "Section_End"
            except:
                continue

            myvocab.REMIID2midi(remi_seq,
                                f"train_mid/{dataset['name']}/mid/{midi}")

            remi_seqs.append(remi_seq)

        pickle.dump(remi_seqs,
                    open(f"data_pkl/{dataset['name']}.pkl", 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    error_midi_count = 0
    midis = []
    for root, dirs, files in os.walk(args.pop909_dir):
        if 'versions' in root:
            continue
        for file in files:
            if file.endswith('.mid'):
                midis.append(os.path.join(root, file))

    time_signature_inThree_idics = [
        34, 62, 102, 107, 152, 173, 176, 203, 215, 231, 254, 280, 307, 328, 369,
        584, 592, 653, 654, 662, 744, 749, 756, 770, 799, 843, 869, 872, 887
    ]
    midis = sorted(midis)

    for label_itr in [1, 2]:
        remi_seqs = []

        for i in tqdm(range(len(midis))):

            id = parse.parse('{}/POP909-Dataset/POP909/{:d}/{:d}.mid',
                             midis[i])[-1]
            if id in time_signature_inThree_idics:
                continue

            remi_seq = myvocab.midi2REMI(midis[i],
                                         include_bridge=False,
                                         verbose=False,
                                         dataset_name="pop909")

            phrase_label_file = f"train_mid/pop909/phrase_label{label_itr}/human_label_{str(id).zfill(3)}.txt"

            try:

                f = open(phrase_label_file, "r")
                text = f.read()
                f.close()
                assert text != ""
                phrase_labels, phrease_bar_nums = myvocab.parse_struct(
                    phrase_label_file)
                assert all(int(x) <= 35 for x in phrease_bar_nums)
                for label in phrase_labels:
                    if label.isupper():
                        assert label in myvocab.phrase_labels
            except:
                continue

            try:
                remi_seq = myvocab.insert_phrase_label(remi_seq,
                                                       phrase_label_file)
            except:
                remi_seq, bar_diff_num = myvocab.fix_phrase_label(
                    remi_seq, phrase_label_file)
                if abs(bar_diff_num) >= 5:
                    error_midi_count += 1
                    continue
                with open('text_dir/error_phrase.txt', 'a') as file:
                    file.write(f"id: {id}, bar_diff_num: {bar_diff_num}\n")

            assert remi_seq.count(myvocab.token2id["Bar"]) == 0
            remi_seq = myvocab.insert_section_start_end(remi_seq)
            remi_seqs.append(remi_seq)
            myvocab.REMIID2midi(remi_seq,
                                f"train_mid/pop909/mid/{str(id).zfill(3)}.mid")

        pickle.dump(remi_seqs,
                    open(f"data_pkl/pop909_{label_itr}.pkl", 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
