import pickle
from tqdm import tqdm
import argparse

import sys
import os
from theme_preprocess.vocab import Vocab as ThemeVocab
from inference import inference
from mymodel import myLM

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)
from vocab import Vocab

from reward_experiment import *
from utils import set_seed

myvocab = Vocab()
mythemevocab = ThemeVocab()

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


def convert_remi_seq(remi_seq):
    remi_seq2 = []
    for x in remi_seq:
        if mythemevocab.id2token[x] in myvocab.token2id:
            remi_seq2.append(myvocab.token2id[mythemevocab.id2token[x]])
    return remi_seq2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument('--sample_num', type=int, default=600)
    parser.add_argument('--log_name', type=str)
    parser.add_argument('--max_len',
                        type=int,
                        default=512,
                        help='number of tokens to predict')
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--use_pkl', action='store_true')
    args = parser.parse_args()

    set_seed(0)
    if args.mode == "experiment1":
        model = myLM(mythemevocab.n_tokens,d_model=256,num_encoder_layers=6,xorpattern=[0,0,0,1,1,1]).cuda()
        model.load_state_dict(torch.load("ckpt/model_ep2311.pt"))

        experiment = RewardExperimentNoPhBc(f'text_dir/{args.log_name}.txt')

        with open("data_pkl/experiment_input1.pkl", "rb") as f:
            input1 = pickle.load(f)

        if args.use_pkl:
            with open(f"data_pkl/{args.log_name}_sampled_remi_seqs.pkl",
                      "rb") as f:
                sampled_remi_seqs = pickle.load(f)
        else:
            sampled_remi_seqs = []

        for i in tqdm(range(args.sample_num)):
            input_data = input1[i % len(input1)]

            if not args.use_pkl:
                while True:
                    sampled_remi_seq = inference(model=model,
                                            theme_seq = input_data["encoder_input_seq"],
                                            prompt = input_data["decoder_input_seq"],
                                            n_bars=input_data["bar_num"],
                                            t=args.t,
                                            p=args.p)
                    if sampled_remi_seq is not None:
                        break

                sampled_remi_seqs.append({
                    "name": input_data["name"],
                    "seq": sampled_remi_seq
                })
            else:
                sampled_remi_seq = sampled_remi_seqs[i]["seq"]            

            sampled_remi_seq = convert_remi_seq(sampled_remi_seq)
            sampled_remi_seq = fix_ph_bc_label(sampled_remi_seq, input_data["phrase_labels"])

            experiment.reward_experiment_fun(sampled_remi_seq)

        if not args.use_pkl:
            with open(f"data_pkl/{args.log_name}_sampled_remi_seqs.pkl",
                        "wb") as handle:
                pickle.dump(sampled_remi_seqs,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)