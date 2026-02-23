import pickle
import numpy as np
import pretty_midi
import librosa
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import os

project_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
))

from reward_experiment import *

from vocab import Vocab

myvocab = Vocab()

import sys

theme_dir = os.path.join(project_dir, 'theme-based_generation')
sys.path.append(theme_dir)
from theme_preprocess.vocab import Vocab as ThemeVocab

mythemevocab = ThemeVocab()


def pad_and_combine_rolls(roll1, roll2):
    max_len = max(roll1.shape[1], roll2.shape[1])

    padded_roll1 = np.pad(roll1, ((0, 0), (0, max_len - roll1.shape[1])),
                          mode='constant')
    padded_roll2 = np.pad(roll2, ((0, 0), (0, max_len - roll2.shape[1])),
                          mode='constant')

    return padded_roll1, padded_roll2


def plot_phrase_based_generation_piano_roll(pm, highlighted_bars, ax):
    fs = 100
    start_pitch = 24 + 12
    end_pitch = 84

    melody_roll = pm.instruments[0].get_piano_roll(fs=fs)
    piano_roll = pm.instruments[1].get_piano_roll(fs=fs)

    melody_roll[melody_roll > 0] = 2
    piano_roll[piano_roll > 0] = 1

    melody_roll, piano_roll = pad_and_combine_rolls(melody_roll, piano_roll)
    roll = melody_roll + piano_roll
    roll = roll[start_pitch:end_pitch]
    roll[roll > 2] = 2

    a1 = roll[:, highlighted_bars[0] * 200:highlighted_bars[1] * 200]
    a1[a1 == 0] = 3

    cmap = ListedColormap(['white', 'grey', 'magenta', 'lavenderblush'])

    librosa.display.specshow(roll,
                             hop_length=1,
                             sr=fs,
                             y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch),
                             cmap=cmap,
                             ax=ax)

    bar_length = 200
    num_bars = roll.shape[1] // bar_length
    bar_ticks = np.arange(0, roll.shape[1], bar_length)
    bar_ticks = bar_ticks[:-1]
    bar_labels = [f'{i+1}' for i in range(len(bar_ticks))]

    ax.set_xticks(bar_ticks)
    ax.set_xticklabels(bar_labels)
    ax.set_xlabel('Bar')
    ax.set_ylabel('Pitch')
    ax.tick_params(axis='y', which='minor', left=False)

    for t in range(0, 16, 4):
        ax.axvline(x=t * 200, color='black', linestyle='-', linewidth=0.8)


def plot_prompt_based_generation_piano_roll(pm, ax):
    fs = 100
    start_pitch = 24 + 12
    end_pitch = 84

    melody_roll = pm.instruments[0].get_piano_roll(fs=fs)
    piano_roll = pm.instruments[1].get_piano_roll(fs=fs)

    melody_roll[melody_roll > 0] = 2
    piano_roll[piano_roll > 0] = 1

    melody_roll, piano_roll = pad_and_combine_rolls(melody_roll, piano_roll)
    roll = melody_roll + piano_roll
    roll = roll[start_pitch:end_pitch]
    roll[roll > 2] = 2

    cmap = ListedColormap(['white', 'grey', 'magenta'])

    librosa.display.specshow(roll,
                             hop_length=1,
                             sr=fs,
                             y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch),
                             cmap=cmap,
                             ax=ax)

    bar_length = 200
    num_bars = roll.shape[1] // bar_length
    bar_ticks = np.arange(0, roll.shape[1], bar_length)
    bar_ticks = bar_ticks[:-1]
    bar_labels = [f'{i+1}' for i in range(len(bar_ticks))]

    ax.set_xticks(bar_ticks)
    ax.set_xticklabels(bar_labels)
    ax.set_xlabel('Bar')
    ax.set_ylabel('Pitch')
    ax.tick_params(axis='y', which='minor', left=False)


def plot_theme_based_generation_piano_roll(pm, highlighted_bars, ax):
    fs = 100
    start_pitch = 24 + 12
    end_pitch = 84

    melody_roll = pm.instruments[0].get_piano_roll(fs=fs)
    piano_roll = pm.instruments[1].get_piano_roll(fs=fs)

    melody_roll[melody_roll > 0] = 2
    piano_roll[piano_roll > 0] = 1

    melody_roll, piano_roll = pad_and_combine_rolls(melody_roll, piano_roll)
    roll = melody_roll + piano_roll
    roll = roll[start_pitch:end_pitch]
    roll[roll > 2] = 2

    for bar in highlighted_bars:
        a1 = roll[:, bar * 200:(bar + 2) * 200]
        a1[a1 == 0] = 3

    cmap = ListedColormap(['white', 'grey', 'magenta', 'lavenderblush'])

    librosa.display.specshow(roll,
                             hop_length=1,
                             sr=fs,
                             y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch),
                             cmap=cmap,
                             ax=ax)

    bar_length = 200
    num_bars = roll.shape[1] // bar_length
    bar_ticks = np.arange(0, roll.shape[1], bar_length)
    bar_labels = [f'{i+1}' for i in range(len(bar_ticks))]

    ax.set_xticks(bar_ticks)
    ax.set_xticklabels(bar_labels)
    ax.set_xlabel('Bar')
    ax.set_ylabel('Pitch')
    ax.tick_params(axis='y', which='minor', left=False)

    for bar in highlighted_bars:
        ax.axvline(x=(bar + 0) * 200,
                   color='black',
                   linestyle='-',
                   linewidth=0.8)
        ax.axvline(x=(bar + 2) * 200,
                   color='black',
                   linestyle='-',
                   linewidth=0.8)


# Prompt
with open(
        os.path.join(project_dir, "prompt-based_generation/", "all_datasets",
                     "data_pkl", "REMI-39_sampled_remi_seqs.pkl"), "rb") as f:
    remi_seqs = pickle.load(f)

seq = remi_seqs[128]['seq']
temp_file_name = '/tmp/' + str(uuid.uuid4()) + '.txt'
experiment = RewardExperimentNoPhBc(temp_file_name)
experiment.reward_experiment_fun(seq)

print(experiment.rewards["melody"][0], experiment.rewards["chord_progression"][0])
myvocab.REMIID2midi(seq, "temp.mid", tempo=120)

pm = pretty_midi.PrettyMIDI('temp.mid')
fig, ax = plt.subplots(figsize=(12, 2.5))
plot_prompt_based_generation_piano_roll(pm, ax=ax)
plt.tight_layout()
plt.savefig("figure_dir/prompt_based.png")
plt.close(fig)


# Theme

with open(
        os.path.join(project_dir, "theme-based_generation/", "data_pkl",
                     "theme_sampled_remi_seqs.pkl"), "rb") as f:
    remi_seqs = pickle.load(f)


def convert_remi_seq(remi_seq):
    remi_seq2 = []
    for x in remi_seq:
        if mythemevocab.id2token[x] in myvocab.token2id:
            remi_seq2.append(myvocab.token2id[mythemevocab.id2token[x]])
    return remi_seq2

def fix_ph_bc_label(words, input_phrase_labels):
    phrase_label_idx = 0
    bar_countdown = 1
    i = 0
    phrase_label = None
    while i < len(words):
        if "Bar" in myvocab.id2token[words[i]]:
            if bar_countdown == 1:
                bar_countdown = int(input_phrase_labels[phrase_label_idx][1:])
                words[i] = myvocab.token2id["Bar_" + str(bar_countdown)]
                phrase_label = myvocab.token2id["Phrase_" + input_phrase_labels[phrase_label_idx][0]]
                words.insert(i + 1, phrase_label)
                phrase_label_idx += 1
            else:
                bar_countdown = bar_countdown - 1
                words[i] = myvocab.token2id["Bar_" + str(bar_countdown)]
                words.insert(i + 1, phrase_label)
        i += 1

    assert bar_countdown == 1 and phrase_label_idx == len( input_phrase_labels) 
    return words


seq = remi_seqs[0]['seq']
seq = convert_remi_seq(seq)
seq = fix_ph_bc_label(seq, ['A16'])
    
experiment = RewardExperimentNoPhBc(temp_file_name)
experiment.reward_experiment_fun(seq)
print(experiment.rewards["melody"][0], experiment.rewards["chord_progression"][0])

mythemevocab.REMIID2midi(remi_seqs[0]['seq'], "temp.mid")
pm = pretty_midi.PrettyMIDI('temp.mid')
fig, ax = plt.subplots(figsize=(12, 2.5))
plot_theme_based_generation_piano_roll(pm, [0, 4], ax=ax)
plt.tight_layout()
plt.savefig("figure_dir/theme_based.png")
plt.close(fig)

with open(
        os.path.join(project_dir, "phrase-based_generation/", "ph_bc",
                     "data_pkl", "ph_bc-39_sampled_remi_seqs.pkl"), "rb") as f:
    remi_seqs = pickle.load(f)

seq=remi_seqs[33+32*3]['seq']
myvocab.REMIID2midi(seq, "temp.mid", tempo=120)
pm = pretty_midi.PrettyMIDI('temp.mid')

experiment = RewardExperimentPhBc(temp_file_name)
experiment.reward_experiment_fun(seq)
print(experiment.rewards["melody"][0], experiment.rewards["chord_progression"][0])


fig, ax = plt.subplots(figsize=(12, 2.5))
plot_phrase_based_generation_piano_roll(pm, [0, 8], ax=ax)
plt.tight_layout()
plt.savefig("figure_dir/phrase_based_0.png")
plt.close(fig)


seq=remi_seqs[0]['seq']
experiment = RewardExperimentPhBc(temp_file_name)
experiment.reward_experiment_fun(seq)
print(experiment.rewards["melody"][0], experiment.rewards["chord_progression"][0])


myvocab.REMIID2midi(seq, "temp.mid", tempo=120)
pm = pretty_midi.PrettyMIDI('temp.mid')
fig, ax = plt.subplots(figsize=(12, 2.5))
plot_phrase_based_generation_piano_roll(pm, [4, 12], ax=ax)
plt.tight_layout()
plt.savefig("figure_dir/phrase_based_1.png")
plt.close(fig)
