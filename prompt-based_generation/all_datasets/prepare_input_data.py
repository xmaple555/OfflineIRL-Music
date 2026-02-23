import pickle

import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from vocab import Vocab

myvocab = Vocab()


def experiment1_data():
    input_datas = []

    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "experiment_input1.pkl"), "rb") as f:
        input_datas = pickle.load(f)

    for data in input_datas:
        data['input_seq'] = myvocab.remove_ph_bc(data['input_seq'])

        bar_num = 0
        for ph in data['phrase_labels']:
            bar_num += int(ph[1:])
        data['bar_num'] = bar_num

        data['phrase_labels'] = [f"A{bar_num}"]


    with open("data_pkl/experiment_input1.pkl", "wb") as handle:
        pickle.dump(input_datas, handle, protocol=pickle.HIGHEST_PROTOCOL)


experiment1_data()
