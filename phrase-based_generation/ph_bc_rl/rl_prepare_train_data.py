import pickle
from tqdm import tqdm
import argparse

import sys
import os
project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)

from vocab import Vocab
from reward_experiment import *
from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len',
                        type=int,
                        default=2048,
                        help='number of tokens to predict')
    args = parser.parse_args()

    myvocab = Vocab()

    datasets = [
        {
            "name": "pop909_1",
            "val_data_num": 20,
            "keep_ratio": 0.5
        },
        {
            "name": "pop909_2",
            "val_data_num": 20,
            "keep_ratio": 0.5
        },
        {
            "name": "nottingham",
            "val_data_num": 20,
            "keep_ratio": 0.5
        },
        {
            "name": "wikifonia",
            "val_data_num": 20,
            "keep_ratio": 0.5
        },
        {
            "name": "theorytab",
            "val_data_num": 100,
            "keep_ratio": 0.5
        },
    ]

    train_dataset = []

    with open("../ph_bc/data_pkl/train_full_seqs.pkl", "rb") as f:
        train_full_seqs = pickle.load(f)

    names = [
        'pop909_1',
        'pop909_2',
        'nottingham',
        'wikifonia',
        'theorytab',
    ]
    experiments = [
        RewardExperimentPhBc(f'text_dir/train_{x}.txt')
        for x in names
    ]

    train_seqs = {
        "pop909_1": [],
        "pop909_2": [],
        "nottingham": [],
        "wikifonia": [],
        "theorytab": []
    }
    for data in tqdm(train_full_seqs):
        if data["name"] not in names:
            continue

        remi_seq = data["seq"]

        train_seqs[data["name"]].append(remi_seq.copy())
        experiments[names.index(
            data["name"])].reward_experiment_fun(remi_seq)

    for i in range(len(experiments)):
        rewards = np.array([
            experiments[i].rewards["melody"],
            experiments[i].rewards["chord_progression"],
        ])
        rewards = rewards.transpose()

        rewards = (rewards - rewards.mean(axis=0)) / rewards.std(axis=0)
        rewards = rewards.mean(axis=1)
        rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())

        rewards2 = rewards.copy()
        rewards2.sort()
        rewards2 = rewards2[::-1]
        threshold_reard = rewards2[
            int(len(rewards2) * datasets[i]["keep_ratio"]) - 1]

        name = names[i]
        assert len(train_seqs[name]) == len(rewards)
        for j in range(len(train_seqs[name])):
            if rewards[j] < threshold_reard:
                continue

            seq = train_seqs[name][j]
            preprocessed_data = myvocab.preprocessREMI(
                seq,
                max_seq_len=args.max_len,
            )

            for k in range(len(preprocessed_data["tgt_segments"])):
                tgt = preprocessed_data["tgt_segments"][k]
                train_dataset.append({
                    "tgt": tgt,
                    "reward": rewards[j],
                })

    with open(f"data_pkl/rl_train_{args.max_len}.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
