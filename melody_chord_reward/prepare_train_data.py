import pickle
from tqdm import tqdm

import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from vocab import Vocab

myvocab = Vocab()

max_melody_seq_len = 1024


def prepare_melody_train_data():

    ph_bc_melody_train_dataset = []
    ph_bc_melody_test_dataset = []
    ph_bc_melody_sampled_dataset = []

    no_ph_bc_melody_train_dataset = []
    no_ph_bc_melody_test_dataset = []
    no_ph_bc_melody_sampled_dataset = []

    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "train_full_seqs.pkl"), "rb") as f:
        train_full_seqs = pickle.load(f)
    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "test_full_seqs.pkl"), "rb") as f:
        test_full_seqs = pickle.load(f)
    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "ph_bc-39_sampled_remi_seqs.pkl"), "rb") as f:
        sampled_seqs = pickle.load(f)

    for_loop_items = [[
        ph_bc_melody_train_dataset, no_ph_bc_melody_train_dataset,
        train_full_seqs
    ], [
        ph_bc_melody_test_dataset, no_ph_bc_melody_test_dataset, test_full_seqs
    ],
                      [
                          ph_bc_melody_sampled_dataset,
                          no_ph_bc_melody_sampled_dataset, sampled_seqs
                      ]]

    for pc_ph_dataset, no_pc_ph_dataset, full_seqs in for_loop_items:

        for data in tqdm(full_seqs, desc="Processing Melody Data"):
            seq = data["seq"]
            section_start_idics = []
            section_end_idics = []
            for i in range(len(seq)):
                if myvocab.id2token[seq[i]] == "Section_Start":
                    section_start_idics.append(i)
                elif myvocab.id2token[seq[i]] == "Section_End":
                    section_end_idics.append(i)

            assert len(section_start_idics) == len(section_end_idics)
            sections = []
            for i in range(len(section_start_idics)):
                sections.append(
                    seq[section_start_idics[i]:section_end_idics[i] + 1])

            for section in sections:
                melody_note_count = 0
                for i in range(len(section)):
                    if "Note-On-MELODY" in myvocab.id2token[section[i]]:
                        melody_note_count += 1

                if melody_note_count < 8:
                    continue

                melody_section = myvocab.melody_to_interval(section)
                for x in range(0, len(melody_section), max_melody_seq_len):
                    pc_ph_dataset.append(melody_section[x:x +
                                                        max_melody_seq_len + 1])

                melody_section = myvocab.remove_ph_bc(melody_section.copy())
                for x in range(0, len(melody_section), max_melody_seq_len):
                    no_pc_ph_dataset.append(
                        melody_section[x:x + max_melody_seq_len + 1])

    with open(f"ph_bc/data_pkl/melody_train.pkl", "wb") as f:
        pickle.dump(ph_bc_melody_train_dataset, f)
    with open(f"ph_bc/data_pkl/melody_test.pkl", "wb") as f:
        pickle.dump(ph_bc_melody_test_dataset, f)
    with open(f"ph_bc/data_pkl/melody_sampled.pkl", "wb") as f:
        pickle.dump(ph_bc_melody_sampled_dataset, f)

    with open(f"no_ph_bc/data_pkl/melody_train.pkl", "wb") as f:
        pickle.dump(no_ph_bc_melody_train_dataset, f)
    with open(f"no_ph_bc/data_pkl/melody_test.pkl", "wb") as f:
        pickle.dump(no_ph_bc_melody_test_dataset, f)
    with open(f"no_ph_bc/data_pkl/melody_sampled.pkl", "wb") as f:
        pickle.dump(no_ph_bc_melody_sampled_dataset, f)


def prepare_chord_train_data():

    ph_bc_chord_train_dataset = []
    ph_bc_chord_test_dataset = []
    ph_bc_chord_sampled_dataset = []

    no_ph_bc_chord_train_dataset = []
    no_ph_bc_chord_test_dataset = []
    no_ph_bc_chord_sampled_dataset = []

    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "train_full_seqs.pkl"), "rb") as f:
        train_full_seqs = pickle.load(f)
    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "test_full_seqs.pkl"), "rb") as f:
        test_full_seqs = pickle.load(f)
    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "ph_bc-39_sampled_remi_seqs.pkl"), "rb") as f:
        sampled_seqs = pickle.load(f)

    for_loop_items = [[
        ph_bc_chord_train_dataset, no_ph_bc_chord_train_dataset, train_full_seqs
    ], [ph_bc_chord_test_dataset, no_ph_bc_chord_test_dataset, test_full_seqs],
                      [
                          ph_bc_chord_sampled_dataset,
                          no_ph_bc_chord_sampled_dataset, sampled_seqs
                      ]]

    for ph_bc_dataset, no_ph_bc_dataset, full_seqs in for_loop_items:
        for data in tqdm(full_seqs, desc="Processing Chord Data"):

            seq = data["seq"]

            seq = myvocab.insert_chord_label(seq)
            seq = myvocab.extract_chord_label(seq)
            ph_bc_dataset.append(seq)

            seq = myvocab.remove_ph_bc(seq.copy())
            no_ph_bc_dataset.append(seq)

    with open(f"ph_bc/data_pkl/chord_train.pkl", "wb") as f:
        pickle.dump(ph_bc_chord_train_dataset, f)
    with open(f"ph_bc/data_pkl/chord_test.pkl", "wb") as f:
        pickle.dump(ph_bc_chord_test_dataset, f)
    with open(f"ph_bc/data_pkl/chord_sampled.pkl", "wb") as f:
        pickle.dump(ph_bc_chord_sampled_dataset, f)

    with open(f"no_ph_bc/data_pkl/chord_train.pkl", "wb") as f:
        pickle.dump(no_ph_bc_chord_train_dataset, f)
    with open(f"no_ph_bc/data_pkl/chord_test.pkl", "wb") as f:
        pickle.dump(no_ph_bc_chord_test_dataset, f)
    with open(f"no_ph_bc/data_pkl/chord_sampled.pkl", "wb") as f:
        pickle.dump(no_ph_bc_chord_sampled_dataset, f)


if __name__ == '__main__':
    prepare_melody_train_data()
    prepare_chord_train_data()
