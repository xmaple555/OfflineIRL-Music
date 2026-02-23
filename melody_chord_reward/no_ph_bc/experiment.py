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
from utils import set_seed

myvocab = Vocab()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    set_seed(0)
    if args.mode == 'test_expert_dataset_reward':
        with open(
                os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                            "test_full_seqs.pkl"), "rb") as f:
            test_full_seqs = pickle.load(f)

        names = ['theorytab', 'wikifonia', 'nottingham', 'pop909_1', 'pop909_2']
        experiments = [
            RewardExperimentNoPhBc2(f'text_dir/test_{x}.txt') for x in names
        ]

        for data in tqdm(test_full_seqs):
            remi_seq = data["seq"]

            experiments[names.index(
                data["name"])].reward_experiment_fun(remi_seq)
