import numpy as np
from utils import chord_recognition
import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
import parse
import re


class Vocab(object):

    def __init__(self, melody_interval_max=12):

        self.token2id = {}
        self.id2token = {}

        # split each beat into 4 subbeats
        self.q_beat = 4

        # dictionary for matching token ID to name and the other way around.
        self.token2id = {}
        self.id2token = {}

        # midi pitch number : 1 ~ 127 (highest pitch)
        self._pitch_bins = np.arange(start=1, stop=128)

        # duration tokens 1~64 of self.q_beat
        self._duration_bins = np.arange(start=1, stop=self.q_beat * 16 + 1)

        # position(subbeat) tokens 0~15, indicate the relative position with in a bar
        self._position_bins = np.arange(start=0, stop=16)

        self.n_tokens = 0

        self.tracks = ["MELODY", "PIANO"]

        self.phrase_labels = ['A', 'B', 'C', 'D', 'X', 'E', 'n']

        self.bar_countdown = 35

        self.phrase_labels = ['A', 'B', 'C', 'D', 'X', 'E', 'n']

        self.chord_roots = [
            'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'
        ]
        self.chord_qualities = [
            'maj',
            'min',
            'aug',
            'dim',
            'sus4',
            'sus4(b7)',
            'sus4(b7,9)',
            'sus2',
            '7',
            'maj7',
            'min7',
            'minmaj7',
            'maj6',
            'min6',
            '9',
            'maj9',
            'min9',
            '7(#9)',
            'maj6(9)',
            'min6(9)',
            'maj(9)',
            'min(9)',
            'maj(11)',
            'min(11)',
            '11',
            'maj9(11)',
            'min11',
            '13',
            'maj13',
            'min13',
            'dim7',
            'hdim7',
        ]

        self.melody_interval_max = melody_interval_max

        self.n_tokens = 0

        self.build()

    def build(self):
        """build our vocab
        """
        self.token2id = {}
        self.id2token = {}

        self.n_tokens = 0

        self.token2id['padding'] = 0
        self.n_tokens += 1

        # Note related tokens==================================================================
        # Create Note-On tokens for each track
        for track in self.tracks:
            # Note-On
            for i in self._pitch_bins:
                self.token2id['Note-On-{}_{}'.format(track, i)] = self.n_tokens
                self.n_tokens += 1

        # Create Note-Duration tokens for each track
        for track in self.tracks:
            # Note-Duration
            for note_dur in self._duration_bins:
                self.token2id['Note-Duration-{}_{}'.format(
                    track, note_dur)] = self.n_tokens
                self.n_tokens += 1

        # Metric related tokens==================================================================

        # Positions
        for pos in self._position_bins:
            self.token2id['Position_{}'.format(pos)] = self.n_tokens
            self.n_tokens += 1

        # Bar
        self.token2id['Bar'] = self.n_tokens
        self.n_tokens += 1
        for i in range(1, self.bar_countdown + 1):
            self.token2id[f'Bar_{i}'] = self.n_tokens
            self.n_tokens += 1

        self.token2id['Section_Start'] = self.n_tokens
        self.n_tokens += 1
        self.token2id['Section_End'] = self.n_tokens
        self.n_tokens += 1

        for phrase_label in self.phrase_labels:
            self.token2id[f"Phrase_{phrase_label}"] = self.n_tokens
            self.n_tokens += 1

        for chord_root in self.chord_roots:
            for chord_quality in self.chord_qualities:
                self.token2id[
                    f"Chord_{chord_root}:{chord_quality}"] = self.n_tokens
                self.n_tokens += 1

        self.token2id["Chord_N"] = self.n_tokens
        self.n_tokens += 1

        self.token2id["Melody_Interval_0"] = self.n_tokens
        self.n_tokens += 1

        for i in range(1, self.melody_interval_max):
            self.token2id[f"Melody_Interval_{i}"] = self.n_tokens
            self.n_tokens += 1

        self.token2id[
            f"Melody_Interval_{self.melody_interval_max}+"] = self.n_tokens
        self.n_tokens += 1

        for i in range(1, self.melody_interval_max):
            self.token2id[f"Melody_Interval_-{i}"] = self.n_tokens
            self.n_tokens += 1

        self.token2id[
            f"Melody_Interval_-{self.melody_interval_max}+"] = self.n_tokens
        self.n_tokens += 1

        for w, v in self.token2id.items():
            self.id2token[v] = w

        self.n_tokens = len(self.token2id)

    def preprocessREMI(self, remi_sequence, max_seq_len):

        tgt_segments = []
        for x in range(0, len(remi_sequence), max_seq_len):
            tgt_segments.append(remi_sequence[x:x + max_seq_len + 1])

        return {
            "tgt_segments": tgt_segments,
        }

    def midi2REMI(self,
                  midi_path,
                  dataset_name,
                  quantize=True,
                  trim_intro=True,
                  trim_outro=True,
                  include_bridge=False,
                  verbose=False):

        assert dataset_name in [
            "theorytab", "wikifonia", "nottingham", "pop909", "musescore"
        ]

        midi_obj = mid_parser.MidiFile(midi_path)
        assert len(midi_obj.instruments) > 0

        if dataset_name == "wikifonia" or dataset_name == "nottingham":
            assert len(midi_obj.instruments) == 2

            midi_obj.instruments[0].name = "MELODY"
            midi_obj.instruments[1].name = "PIANO"

        elif dataset_name == "theorytab":
            assert len(midi_obj.instruments) == 3

            midi_obj.instruments[0].name = "MELODY"
            midi_obj.instruments[1].name = "PIANO"
            del midi_obj.instruments[2]

        # calculate the min step (in ticks) for REMI representation
        min_step = midi_obj.ticks_per_beat * 4 / 16

        # quantize
        if quantize:
            for i in range(len(midi_obj.instruments)):
                for n in range(len(midi_obj.instruments[i].notes)):
                    midi_obj.instruments[i].notes[n].start = int(
                        int(midi_obj.instruments[i].notes[n].start / min_step) *
                        min_step)
                    midi_obj.instruments[i].notes[n].end = int(
                        int(midi_obj.instruments[i].notes[n].end / min_step) *
                        min_step)

        # parsing notes in each tracks (ignore BRIDGE)
        notesAndtempos = []

        midi_obj.instruments[0].notes = sorted(midi_obj.instruments[0].notes,
                                               key=lambda x: x.start)
        # add notes
        melody_start = sorted(midi_obj.instruments[0].notes,
                              key=lambda x: x.start)[0].start
        melody_end = sorted(midi_obj.instruments[0].notes,
                            key=lambda x: x.start)[-1].end

        for i in range(len(midi_obj.instruments)):
            if not include_bridge and midi_obj.instruments[i].name == "BRIDGE":
                continue

            notes = midi_obj.instruments[i].notes
            for n in notes:
                # assert (trim_intro and n.start>=melody_start or not trim_intro)
                if trim_intro and n.start >= melody_start or not trim_intro:
                    if trim_outro and n.start <= melody_end or not trim_outro:

                        assert n.start - melody_start >= 0
                        notesAndtempos.append({
                            "priority":
                                i + 1,
                            "priority_1":
                                n.pitch,
                            "start_tick":
                                n.start - melody_start,
                            "obj_type":
                                "Note-{}".format(midi_obj.instruments[i].name),
                            "obj":
                                n
                        })

        notesAndtempos = sorted(
            notesAndtempos,
            key=lambda x: (x["start_tick"], x["priority"], -x["priority_1"]))

        melody_start = 0
        # group
        bar_segments = []
        bar_ticks = midi_obj.ticks_per_beat * 4
        if verbose:
            print("Bar tick length: {}".format(bar_ticks))

        for bar_start_tick in range(0, notesAndtempos[-1]["start_tick"],
                                    bar_ticks):
            if verbose:
                print("Bar {} at tick: {}".format(bar_start_tick // bar_ticks,
                                                  bar_start_tick))
            bar_end_tick = bar_start_tick + bar_ticks
            current_bar = []
            bar_objs = list(
                filter(
                    lambda x: x["start_tick"] >= bar_start_tick and x[
                        "start_tick"] < bar_end_tick, notesAndtempos))
            bar_objs.insert(0, {"start_tick": -1})

            if trim_intro and bar_start_tick + bar_ticks < melody_start:
                if verbose:
                    print("into trimmed")
                continue

            current_bar.append("Bar")

            for i, obj in enumerate(bar_objs):
                if obj["start_tick"] == -1:
                    continue
                if not obj["start_tick"] == bar_objs[i - 1]["start_tick"]:
                    # insert Position Event
                    pos = (obj["start_tick"] - bar_start_tick
                          ) / midi_obj.ticks_per_beat * self.q_beat
                    pos_index = np.argmin(
                        abs(pos -
                            self._position_bins))  # use the closest position
                    pos = self._position_bins[pos_index]
                    current_bar.append("Position_{}".format(pos))

                if obj["obj_type"].startswith("Note"):
                    track_name = obj["obj_type"].split('-')[1].upper()
                    # add pitch
                    current_bar.append("Note-On-{}_{}".format(
                        track_name, obj["obj"].pitch))
                    # add duration
                    dur = (obj["obj"].end - obj["obj"].start
                          ) / midi_obj.ticks_per_beat * self.q_beat
                    dur_index = np.argmin(
                        abs(dur -
                            self._duration_bins))  # use the closest position
                    dur = self._duration_bins[dur_index]
                    current_bar.append("Note-Duration-{}_{}".format(
                        track_name, dur))

                else:
                    current_bar.append(obj["obj_type"])

            bar_segments.extend(current_bar)

        output_ids = [self.token2id[x] for x in bar_segments]

        return output_ids

    def parse_struct(self, struct_file):
        f = open(struct_file, "r")
        line = f.readline().strip()
        f.close()
        line.replace("\n", "")
        labels = re.split("[0-9]+", line)
        nums = re.split("[a-zA-Z]+", line)
        if "" in labels:
            labels.remove("")
        if "" in nums:
            nums.remove("")

        if labels[0] == "i":
            labels.pop(0)
            nums.pop(0)

        if labels[-1] == "o":
            labels.pop(-1)
            nums.pop(-1)

        assert len(labels) == len(nums)

        labels = [
            'n' if re.match('[a-z]', label) else label for label in labels
        ]
        return labels, nums

    def insert_phrase_label(self, remi_sequence, struct_file):
        phrase_labels, phrease_bar_nums = self.parse_struct(struct_file)
        i = 0
        j = 0
        current_phrase = phrase_labels[j]
        current_phrase_bar_num = int(phrease_bar_nums[j])
        while i < len(remi_sequence):
            event = self.id2token[remi_sequence[i]]
            if event == 'Bar':
                if current_phrase_bar_num == 0:
                    j += 1
                    current_phrase = phrase_labels[j]
                    current_phrase_bar_num = int(phrease_bar_nums[j])
                remi_sequence[i] = self.token2id[
                    f"Bar_{current_phrase_bar_num}"]
                i += 1
                remi_sequence.insert(i,
                                     self.token2id[f"Phrase_{current_phrase}"])
                current_phrase_bar_num -= 1
                i += 1
                continue
            i += 1

        assert j == len(phrase_labels) - 1 and current_phrase_bar_num == 0
        return remi_sequence

    def fix_phrase_label(self, remi_sequence, struct_file):

        # for x in remi_sequence[:phrase_label_idices[sum(phrease_bar_nums[:-1]) - 1]]:
        #     token = self.id2token[x]
        #     if 'Bar' in token or 'Phrase' in token:
        #         print(token)

        ## brute_force_fix

        while remi_sequence[-1] == self.token2id['Bar']:
            remi_sequence.pop(-1)

        phrase_labels, phrease_bar_nums = self.parse_struct(struct_file)
        phrease_bar_nums = [int(x) for x in phrease_bar_nums]
        phrase_label_idices = [
            i for i, event in enumerate(remi_sequence)
            if 'Phrase' in self.id2token[event]
        ]
        phrase_label_idices.sort()
        last_phrase_index = phrase_label_idices[-1]

        last_second_phrase_index = 0

        if remi_sequence.count(self.token2id['Bar']) > 0:
            bar_one_count = 0
            for i in range(len(remi_sequence) - 1, 0, -1):
                event = self.id2token[remi_sequence[i]]
                if 'Phrase' in event:
                    assert self.id2token[remi_sequence[i - 1]].split(
                        '_')[0] == 'Bar'
                    target = int(self.id2token[remi_sequence[i -
                                                             1]].split('_')[1])
                    if target == 1:
                        bar_one_count += 1
                        if bar_one_count == 2:
                            last_second_phrase_index = i
                            break

            last_phrase_label = self.id2token[remi_sequence[last_phrase_index]]
            i = len(remi_sequence) - 1
            bar_count = 1

            first_bar_one = False
            while i >= 0 and i > last_second_phrase_index:
                event = self.id2token[remi_sequence[i]]
                if event == 'Bar':
                    assert 'Phrase_' not in self.id2token[remi_sequence[i + 1]]
                    remi_sequence[i] = self.token2id[f"Bar_{bar_count}"]
                    bar_count += 1
                    remi_sequence.insert(i + 1,
                                         self.token2id[last_phrase_label])
                    i -= 1
                    continue

                elif 'Phrase_' in event:
                    assert last_phrase_label == event
                    assert self.id2token[remi_sequence[i - 1]].split(
                        '_')[0] == 'Bar'
                    target = int(self.id2token[remi_sequence[i -
                                                             1]].split('_')[1])
                    if target == bar_count:
                        break

                    if not first_bar_one and target == 1:
                        first_bar_one = True
                    elif first_bar_one and target == 1:
                        break

                    remi_sequence[i - 1] = self.token2id[f"Bar_{bar_count}"]
                    bar_count += 1
                    i -= 2
                    continue

                i -= 1

        else:
            last_phrase_label = self.id2token[remi_sequence[last_phrase_index]]
            for i in range(len(remi_sequence) - 1, 0, -1):
                event = self.id2token[remi_sequence[i]]
                if 'Phrase' in event:
                    assert self.id2token[remi_sequence[i - 1]].split(
                        '_')[0] == 'Bar'
                    target = int(self.id2token[remi_sequence[i -
                                                             1]].split('_')[1])
                    if target == 1:
                        last_second_phrase_index = i
                        break

            assert self.id2token[remi_sequence[last_phrase_index -
                                               1]].split('_')[0] == 'Bar'

            if int(self.id2token[remi_sequence[last_phrase_index -
                                               1]].split('_')[1]) > 1:
                i = len(remi_sequence) - 1
                bar_count = 1

                while i >= 0 and i > last_second_phrase_index:
                    event = self.id2token[remi_sequence[i]]
                    if 'Phrase_' in event:
                        assert last_phrase_label == event
                        assert self.id2token[remi_sequence[i - 1]].split(
                            '_')[0] == 'Bar'
                        target = int(
                            self.id2token[remi_sequence[i - 1]].split('_')[1])
                        if target == bar_count:
                            break

                        if target == 1:
                            break
                        remi_sequence[i - 1] = self.token2id[f"Bar_{bar_count}"]
                        bar_count += 1
                    i -= 1

        bar_diff_num = len([
            i for i, event in enumerate(remi_sequence)
            if 'Bar' in self.id2token[event]
        ]) - sum(phrease_bar_nums)

        assert remi_sequence.count(self.token2id['Bar']) == 0

        # for x in remi_sequence:
        #     token = self.id2token[x]
        #     if 'Bar' in token or 'Phrase' in token:
        #         print(token)
        return remi_sequence, bar_diff_num

    def insert_section_start_end(self, remi_sequence):
        assert self.id2token[remi_sequence[1]] in [
            'Phrase_A', 'Phrase_B', 'Phrase_C', 'Phrase_D', 'Phrase_X',
            'Phrase_E'
        ]
        remi_sequence.insert(0, self.token2id['Section_Start'])
        state = True
        i = 0
        while i < len(remi_sequence):
            event = self.id2token[remi_sequence[i]]
            if state and 'Phrase_n' == event:
                assert 'Bar' in self.id2token[remi_sequence[i - 1]]
                state = False
                remi_sequence.insert(i - 1, self.token2id['Section_End'])
                i += 2
                continue
            elif not state and event in [
                    'Phrase_A', 'Phrase_B', 'Phrase_C', 'Phrase_D', 'Phrase_X',
                    'Phrase_E'
            ]:
                assert 'Bar' in self.id2token[remi_sequence[i - 1]]
                remi_sequence.insert(i - 1, self.token2id['Section_Start'])
                state = True
                i += 2
                continue
            i += 1

        for i in range(len(remi_sequence) - 1, 0, -1):
            if 'Phrase' in self.id2token[remi_sequence[i]]:
                if "Phrase_n" != self.id2token[remi_sequence[i]]:
                    remi_sequence.append(self.token2id['Section_End'])
                break

        return remi_sequence

    def insert_chord_label(self, remi_sequence):
        chords = chord_recognition(self, remi_sequence)
        start_times = []
        chord_names = []
        for line in chords:
            start_times.append(line[0])
            chord_names.append(line[2])

        assert len(start_times) == len(chord_names)

        tempo = 120
        ticks_per_beat = 120
        ticks_per_step = 30
        tick_per_sec = (tempo * ticks_per_beat) / 60

        bar_indices = [
            i for i, event in enumerate(remi_sequence)
            if 'Bar' in self.id2token[event]
        ]

        for i in range(len(start_times)):
            chord = chord_names[i]
            if f'Chord_{chord}' in self.token2id:
                chord_word = self.token2id[f'Chord_{chord}']
                pos = round(start_times[i] * tick_per_sec / ticks_per_step)
                bar_num = pos // 16
                pos = pos % 16

                # target_pos_index = [i for i, event in enumerate(remi_sequence[bar_indices[bar_num]:bar_indices[bar_num+1]]) if 'Position' in self.id2token[event] and int(self.id2token[event].split('_')[1]) == pos]
                # assert len(target_pos_index) >= 1
                # target_pos_index = target_pos_index[0] + bar_indices[bar_num]
                # state = False
                # for j in range(target_pos_index + 1, len(remi_sequence)):
                #     event = self.id2token[remi_sequence[j]]
                #     if 'Position' in event or 'Bar' in event:
                #         break
                #     if 'Note-On-PIANO' in event:
                #         state = True
                #         break
                # assert state
                # remi_sequence.insert(target_pos_index, chord_word)
                # remi_sequence.insert(target_pos_index, self.token2id[f'Position_{pos}'])
                # bar_indices = [k for k, event in enumerate(remi_sequence) if 'Bar' in self.id2token[event]]

                if bar_num + 1 < len(bar_indices):
                    for j in range(bar_indices[bar_num] + 1,
                                   bar_indices[bar_num + 1]):
                        event = self.id2token[remi_sequence[j]]
                        if 'Position' in event and int(
                                event.split('_')[1]) >= pos:
                            remi_sequence.insert(j, chord_word)
                            remi_sequence.insert(
                                j, self.token2id[f'Position_{pos}'])
                            bar_indices = [
                                k for k, event in enumerate(remi_sequence)
                                if 'Bar' in self.id2token[event]
                            ]
                            break
                        elif 'Bar' in event:
                            raise ValueError("should not happen")

                            # remi_sequence.insert(j, chord_word)
                            # remi_sequence.insert(
                            #     j, self.token2id[f'Position_{pos}'])
                            # bar_indices = [
                            #     k for k, event in enumerate(remi_sequence)
                            #     if 'Bar' in self.id2token[event]
                            # ]
                            # break

                elif bar_num + 1 == len(bar_indices):
                    for j in range(bar_indices[bar_num] + 1,
                                   len(remi_sequence)):
                        event = self.id2token[remi_sequence[j]]
                        if 'Position' in event and int(
                                event.split('_')[1]) >= pos:
                            remi_sequence.insert(j, chord_word)
                            remi_sequence.insert(
                                j, self.token2id[f'Position_{pos}'])
                            bar_indices = [
                                k for k, event in enumerate(remi_sequence)
                                if 'Bar' in self.id2token[event]
                            ]
                            break
                        elif 'Bar' in event:
                            raise ValueError("should not happen2")
                            # remi_sequence.insert(j, chord_word)
                            # remi_sequence.insert(
                            #     j, self.token2id[f'Position_{pos}'])
                            # bar_indices = [
                            #     k for k, event in enumerate(remi_sequence)
                            #     if 'Bar' in self.id2token[event]
                            # ]
                            # break
                else:
                    break

        return remi_sequence

    def extract_chord_label(self, remi_sequence):
        remi_sequence2 = []
        for i in range(len(remi_sequence)):
            if "MELODY" in self.id2token[
                    remi_sequence[i]] or "PIANO" in self.id2token[remi_sequence[
                        i]] or 'Section' in self.id2token[remi_sequence[i]]:
                continue
            remi_sequence2.append(remi_sequence[i])

        i = 0
        while i < len(remi_sequence2):
            if "Position" in self.id2token[remi_sequence2[i]]:
                if i + 1 == len(remi_sequence2) or (
                        "Chord" not in self.id2token[remi_sequence2[i + 1]]):
                    remi_sequence2.pop(i)
                    continue
            i += 1
        return remi_sequence2

    def extract_phrase_label(self, remi_sequence, return_indices=False):
        new_phrase_indices = []
        phrase_labels = []
        bar_countdown = 1
        phrase_label = None
        for i in range(len(remi_sequence) - 1):
            event = self.id2token[remi_sequence[i]]
            if "Bar" in event:
                if bar_countdown == 1 and self.id2token[remi_sequence[
                        i + 1]].split('_')[1].isupper():
                    phrase_label = self.id2token[remi_sequence[i +
                                                               1]].split('_')[1]
                    bar_num = event.split('_')[1]
                    phrase_labels.append(phrase_label + bar_num)
                    new_phrase_indices.append(i)

                bar_countdown = int(event.split('_')[1])

        if return_indices:
            return phrase_labels, new_phrase_indices

        return phrase_labels

    def remove_piano_note(self, remi_seq):
        remi_seq2 = []
        for x in remi_seq:
            if not 'PIANO' in self.id2token[x]:
                remi_seq2.append(x)

        i = 0
        while (i + 1) < len(remi_seq2):
            if "Position" in self.id2token[remi_seq2[i]]:
                if "MELODY" not in self.id2token[remi_seq2[i + 1]]:
                    remi_seq2.pop(i)
                    continue
            i += 1

        return remi_seq2

    def melody_to_interval(self, remi_seq):
        remi_seq2 = []
        for x in remi_seq:
            if not 'PIANO' in self.id2token[x]:
                remi_seq2.append(x)

        i = 0
        while (i + 1) < len(remi_seq2):
            if "Position" in self.id2token[remi_seq2[i]]:
                if "MELODY" not in self.id2token[remi_seq2[i + 1]]:
                    remi_seq2.pop(i)
                    continue
            i += 1

        remi_seq = remi_seq2

        last_pitch = None
        for i in range(len(remi_seq)):
            if "Note-On-MELODY" in self.id2token[remi_seq[i]]:
                current_pitch = parse.parse('Note-On-MELODY_{:d}',
                                            self.id2token[remi_seq[i]])[0]
                if last_pitch is None:
                    interval_token = self.token2id[f"Melody_Interval_{0}"]
                else:
                    interval = current_pitch - last_pitch
                    if interval >= self.melody_interval_max:
                        interval_token = self.token2id[
                            f"Melody_Interval_{self.melody_interval_max}+"]
                    elif interval <= -self.melody_interval_max:
                        interval_token = self.token2id[
                            f"Melody_Interval_-{self.melody_interval_max}+"]
                    else:
                        interval_token = self.token2id[
                            f"Melody_Interval_{interval}"]

                remi_seq[i] = interval_token
                last_pitch = current_pitch

        remi_seq2 = [x for x in remi_seq if "Section" not in self.id2token[x]]
        return remi_seq2

    def remove_ph_bc(self, remi_seq):
        i = 0
        while i < len(remi_seq):
            if "Bar_" in self.id2token[remi_seq[i]]:
                remi_seq[i] = self.token2id["Bar"]
                i += 1
            elif "Phrase" in self.id2token[
                    remi_seq[i]] or "Section" in self.id2token[remi_seq[i]]:
                del remi_seq[i]
            else:
                i += 1
        return remi_seq

    def remove_bc(self, remi_seq):
        for i in range(len(remi_seq)):

            if "Bar_" in self.id2token[remi_seq[i]]:
                remi_seq[i] = self.token2id['Bar']
        return remi_seq

    def remove_ph(self, remi_seq):
        i = 0
        while i < len(remi_seq):
            if "Phrase" in self.id2token[remi_seq[i]]:
                del remi_seq[i]
            else:
                i += 1
        return remi_seq

    def REMIID2midi(self, event_ids, midi_path, verbose=False, tempo=120):
        """convert tokens to midi file
        The output midi file will contains 3 tracks:
            MELODY : melodt notes
            PIANO : accompaniment notes
        Args:
            event_ids (list): sequence of tokens
            midi_path (str): the output midi file path
            verbose (bool, optional): print some message. Defaults to False.
        """

        # create midi file
        new_mido_obj = mid_parser.MidiFile()
        new_mido_obj.ticks_per_beat = 120

        # create tracks
        music_tracks = {}
        music_tracks["MELODY"] = ct.Instrument(program=0,
                                               is_drum=False,
                                               name='MELODY')
        music_tracks["PIANO"] = ct.Instrument(program=0,
                                              is_drum=False,
                                              name='PIANO')
        music_tracks["Secction"] = ct.Instrument(program=0,
                                                 is_drum=False,
                                                 name='Secction')

        # all our generated music are 4/4
        new_mido_obj.time_signature_changes.append(
            miditoolkit.TimeSignature(4, 4, 0))

        ticks_per_step = new_mido_obj.ticks_per_beat / self.q_beat

        # convert tokens from id to string
        events = []
        for x in event_ids:
            events.append(self.id2token[x])

        # parsing tokens
        last_tick = 0
        current_bar_anchor = 0

        idx = 0
        first_bar = True
        new_phrase_label = True
        last_phrase_label = None
        current_section_boundary = []

        new_mido_obj.tempo_changes.append(ct.TempoChange(tempo=tempo, time=0))

        while (idx < len(events)):
            if "Bar" in events[idx]:
                if first_bar:
                    current_bar_anchor = 0
                    first_bar = False
                else:
                    current_bar_anchor += new_mido_obj.ticks_per_beat * 4
                idx += 1
                last_tick = current_bar_anchor
            elif events[idx].startswith("Position"):
                pos = int(events[idx].split('_')[1])
                last_tick = pos * ticks_per_step + current_bar_anchor
                idx += 1

            elif events[idx].startswith("Note"):
                track_name = events[idx].split("_")[0].split("-")[2]
                assert track_name in music_tracks
                assert events[idx].startswith("Note-On")
                assert events[idx + 1].startswith("Note-Duration")

                new_note = miditoolkit.Note(
                    velocity=90,
                    pitch=int(events[idx].split("_")[1]),
                    start=int(last_tick),
                    end=int(
                        int(events[idx + 1].split('_')[1]) * ticks_per_step) +
                    int(last_tick))
                music_tracks[track_name].notes.append(new_note)
                idx += 2

            elif events[idx].startswith("Phrase"):
                label = events[idx].split("_")[1]
                num = events[idx - 1].split("_")[1]
                text = f"{label}_{num}"
                if new_phrase_label or last_phrase_label != label:
                    new_mido_obj.markers.append(
                        miditoolkit.Marker(text=text, time=int(last_tick)))
                    last_phrase_label = label
                    new_phrase_label = False
                else:
                    if int(num) == 1:
                        new_phrase_label = True
                idx += 1

            elif events[idx] == "Section_Start":
                assert len(current_section_boundary) == 0
                current_section_boundary.append(last_tick)
                idx += 1

            elif events[idx] == "Section_End":
                assert len(current_section_boundary) == 1
                current_section_boundary.append(last_tick)
                music_tracks["Secction"].notes.append(
                    miditoolkit.Note(velocity=1,
                                     pitch=0,
                                     start=int(current_section_boundary[0]),
                                     end=int(current_section_boundary[1])))
                current_section_boundary = []
                idx += 1

        # add tracks to midi file
        new_mido_obj.instruments.extend(
            [music_tracks[ins] for ins in music_tracks])

        if verbose:
            print("Saving midi to ({})".format(midi_path))

        # save to disk
        new_mido_obj.dump(midi_path)

    def print_seq(self, seq):
        for token in seq:
            print(self.id2token[token])

    def __str__(self):
        """return all tokens

        Returns:
            str: string of all tokens
        """
        ret = ""
        for w, i in self.token2id.items():
            ret = ret + "{} : {}\n".format(w, i)

        for i, w in self.id2token.items():
            ret = ret + "{} : {}\n".format(i, w)

        ret += "\nTotal events #{}".format(len(self.id2token))

        return ret

    def __repr__(self):
        """return string all token

        Returns:
            str: string of sll tokens
        """
        return self.__str__()


if __name__ == '__main__':
    myvocab = Vocab()

    print(myvocab)
