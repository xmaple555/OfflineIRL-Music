import numpy as np
import torch
import random
from exported_midi_chord_recognition.chord_class import ChordClass, NUM_TO_ABS_SCALE
import musthe
import os
import sys
import uuid

sys.path.append('exported_midi_chord_recognition')
from exported_midi_chord_recognition.main import transcribe_cb1000_midi


def number_to_note(number: int) -> tuple:
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES
    assert 0 <= number <= 127
    note = NOTES[number % NOTES_IN_OCTAVE]
    return musthe.Note(note + str(octave))


chord_class = ChordClass()


def chord_to_note(chord: str):
    chord_class.chroma_templates[chord_class.chord_list.index(chord)].copy()

    notes = []
    for i in range(12):
        if chord_class.chroma_templates[chord_class.chord_list.index(
                chord)][i] == 1:
            notes.append(musthe.Note(NUM_TO_ABS_SCALE[i]))
    return notes


def chord_recognition(myvocab, remi_seq):
    file_name = '/tmp/' + str(uuid.uuid4()) + '.mid'
    myvocab.REMIID2midi(remi_seq, file_name)
    result = transcribe_cb1000_midi(file_name)
    os.remove(file_name)

    i = 0
    while True:
        if i >= len(result):
            break
        chord = result[i][2]
        if '/' in chord:
            chord = chord.split('/')[0]
            result[i][2] = chord
        i += 1

    return result


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True