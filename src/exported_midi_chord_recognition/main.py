import os
import sys

src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

from mir import DataEntry
from mir import io
from extractors.midi_utilities import get_valid_channel_count,is_percussive_channel,MidiBeatExtractor
from extractors.rule_based_channel_reweight import midi_to_thickness_and_bass_weights
from midi_chord import ChordRecognition
from chord_class import ChordClass
import numpy as np
from io_new.chordlab_io import ChordLabIO
from io_new.downbeat_io import DownbeatIO
import argparse
import parse
import tqdm
import collections

def process_chord(entry, extra_division):
    '''

    Parameters
    ----------
    entry: the song to be processed. Properties required:
        entry.midi: the pretry midi object
        entry.beat: extracted beat and downbeat
    extra_division: extra divisions to each beat.
        For chord recognition on beat-level, use extra_division=1
        For chord recognition on half-beat-level, use extra_division=2

    Returns
    -------
    Extracted chord sequence
    '''

    midi=entry.midi
    midi.instruments = [instrument for instrument in midi.instruments if instrument.name == 'PIANO']
    beats=midi.get_beats()
    if(extra_division>1):
        beat_interp=np.linspace(beats[:-1],beats[1:],extra_division+1).T
        last_beat=beat_interp[-1,-1]
        beats=np.append(beat_interp[:,:-1].reshape((-1)),last_beat)
    downbeats=midi.get_downbeats()
    j=0
    beat_pos=-2
    beat=[]
    for i in range(len(beats)):
        if(j<len(downbeats) and beats[i]==downbeats[j]):
            beat_pos=1
            j+=1
        else:
            beat_pos=beat_pos+1
        assert(beat_pos>0)
        beat.append([beats[i],beat_pos])
    rec=ChordRecognition(entry,ChordClass())
    weights=midi_to_thickness_and_bass_weights(entry.midi)
    rec.process_feature(weights)
    chord=rec.decode()
    return chord

def transcribe_cb1000_midi(midi_path,output_path = None):
    '''
    Perform chord recognition on a midi
    :param midi_path: the path to the midi file
    :param output_path: the path to the output file
    '''
    entry=DataEntry()
    entry.append_file(midi_path,io.MidiIO,'midi')
    entry.append_extractor(MidiBeatExtractor,'beat')
    result=process_chord(entry,extra_division=2)
    entry.append_data(result,ChordLabIO,'pred')
    if output_path is not None:
        entry.save('pred',output_path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop909_dir", type=str, default='./expert_mid2')
    parser.add_argument("--output_dir", type=str, default='./chord_label')
    args = parser.parse_args()


    midis = []
    for root, dirs, files in os.walk(args.pop909_dir):
        for file in files:
            if file.endswith('.mid'):
                midis.append(os.path.join(root, file))

    chord_statistics = collections.defaultdict(int)
    for midi in tqdm.tqdm(midis):
        id = parse.parse('./expert_mid2/expert_{}.mid',
                         midi)[0]
        
        output_path = os.path.join(args.output_dir, 'chord_'+id+ '.txt')
        result = transcribe_cb1000_midi(midi,output_path)
        for i in range(len(result)):
            chord = result[i][2].split(':')[-1]
            if '/' in chord:
                chord = chord.split('/')[0]
            chord_statistics[chord]+=1

    with open('text_dir/chord_statistics.txt', 'w') as f:
            sorted_chords = sorted(chord_statistics.items(), key=lambda x: x[1], reverse=True)
            for chord, count in sorted_chords:
                f.write(f"{chord}: {count}\n")    

    