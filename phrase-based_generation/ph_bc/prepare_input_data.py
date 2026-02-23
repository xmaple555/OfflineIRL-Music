import pickle

import sys
import os

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from vocab import Vocab

myvocab = Vocab()


def get_input_and_phrase_labels(word_seq):
    first_section_end_idx = word_seq.index(myvocab.token2id["Section_End"])

    first_section_seq = word_seq[:first_section_end_idx + 1]

    phrase_labels = myvocab.extract_phrase_label(first_section_seq)

    input_bar_num = int(phrase_labels[0][1:])
    second_phrase_index = 0
    bar_num = 0
    for i in range(len(first_section_seq)):
        if "Bar" in myvocab.id2token[first_section_seq[i]]:
            bar_num += 1
        if bar_num == input_bar_num + 1:
            second_phrase_index = i
            break

    input_seq = first_section_seq[:second_phrase_index]
    return first_section_seq, input_seq, phrase_labels


def experiment_input_1():
    input_datas = []

    with open("data_pkl/test_full_seqs.pkl", "rb") as f:
        test_full_seqs = pickle.load(f)

    for data in test_full_seqs:
        if data["name"] != "pop909_1" and data["name"] != "pop909_2":
            continue
        first_section_seq, input_seq, phrase_labels = get_input_and_phrase_labels(
            data["seq"])

        if len(phrase_labels) >= 2:
            if phrase_labels[0] == 'A8':
                input_datas.append({
                    "name": data["name"],
                    "input_seq": input_seq,
                    "phrase_labels": ['A8', 'B8', 'B8', 'C8'],
                })

                input_datas.append({
                    "name": data["name"],
                    "input_seq": input_seq,
                    "phrase_labels": ['A8', 'A8', 'B8', 'C8'],
                })

            elif phrase_labels[0] == 'A4':
                input_datas.append({
                    "name": data["name"],
                    "input_seq": input_seq,
                    "phrase_labels": ['A4', 'B4', 'B4', 'C4'],
                })

                input_datas.append({
                    "name": data["name"],
                    "input_seq": input_seq,
                    "phrase_labels": ['A4', 'A4', 'B4', 'C4'],
                })

    with open("data_pkl/experiment_input1.pkl", "wb") as handle:
        pickle.dump(input_datas, handle, protocol=pickle.HIGHEST_PROTOCOL)


experiment_input_1()