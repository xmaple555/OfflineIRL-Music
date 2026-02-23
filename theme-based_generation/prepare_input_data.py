import pickle
import sys
import os
from theme_preprocess.vocab import Vocab as ThemeVocab
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from vocab import Vocab

myvocab = Vocab()
mythemevocab = ThemeVocab()

input_tmp = 120
tmp = mythemevocab._tempo_bins[np.argmin(
    abs(input_tmp - mythemevocab._tempo_bins))]
given_tempo = mythemevocab.token2id["Tempo_{}".format(tmp)]


def experiment1_data():
    input_datas = []

    with open(
            os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                         "experiment_input1.pkl"), "rb") as f:
        input_datas = pickle.load(f)

    for data in input_datas:
        data['input_seq'] = myvocab.remove_ph_bc(data['input_seq'])
        myvocab.REMIID2midi(data['input_seq'], "/tmp/tmp_theme.mid")
        decoder_input_seq = mythemevocab.midi2REMI(
            "/tmp/tmp_theme.mid",
            trim_intro=False,
            trim_outro=False,
            theme_annotations=False,
        )
        decoder_input_seq = [
            x for x in decoder_input_seq
            if not mythemevocab.id2token[x].startswith("Tempo")
        ]

        encoder_input_seq = []
        bar_count = 0
        second_bar_idx = 0

        for i in range(len(decoder_input_seq)):
            x = decoder_input_seq[i]
            if "Bar" == mythemevocab.id2token[x]:
                bar_count += 1
                if bar_count > 2:
                    second_bar_idx = i
                    break
            encoder_input_seq.append(x)

        encoder_input_seq = [
            mythemevocab.token2id["Theme_Start"]
        ] + encoder_input_seq + [mythemevocab.token2id["Theme_End"]]
        decoder_input_seq.insert(second_bar_idx,
                                 mythemevocab.token2id["Theme_End"])
        decoder_input_seq.insert(0, mythemevocab.token2id["Theme_Start"])
        decoder_input_seq = [given_tempo] + decoder_input_seq + [
            mythemevocab.token2id["Theme_Start"]
        ]

        bar_num = 0
        for ph in data['phrase_labels']:
            bar_num += int(ph[1:])
        data['bar_num'] = bar_num
        data['phrase_labels'] = [f"A{bar_num}"]
        data['encoder_input_seq'] = encoder_input_seq
        data['decoder_input_seq'] = decoder_input_seq
        del data["input_seq"]

    with open("data_pkl/experiment_input1.pkl", "wb") as handle:
        pickle.dump(input_datas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    os.remove("/tmp/tmp_theme.mid")


experiment1_data()
