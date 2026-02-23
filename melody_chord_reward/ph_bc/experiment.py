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

theme_dir = os.path.join(project_dir, 'theme-based_generation')
sys.path.append(theme_dir)
from theme_preprocess.vocab import Vocab as ThemeVocab
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
    args = parser.parse_args()

    set_seed(0)

    if args.mode == 'test_expert_dataset_reward':
        with open(
                os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                            "test_full_seqs.pkl"), "rb") as f:
            test_full_seqs = pickle.load(f)

        names = ['theorytab', 'wikifonia', 'nottingham', 'pop909_1', 'pop909_2']
        experiments = [
            RewardExperimentPhBc(f'text_dir/test_{x}.txt') for x in names
        ]

        for data in tqdm(test_full_seqs):
            remi_seq = data["seq"]
            experiments[names.index(
                data["name"])].reward_experiment_fun(remi_seq)


    elif args.mode == 't_test':
        from scipy import stats

        with open(
                os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                            "test_full_seqs.pkl"), "rb") as f:
            test_full_seqs = pickle.load(f)

        temp_file_name = '/tmp/' + str(uuid.uuid4()) + '.txt'
        temp_experiment = RewardExperimentPhBc(temp_file_name)

        for data in tqdm(test_full_seqs):
            if data["name"] == "pop909_1":
                remi_seq = data["seq"]
                temp_experiment.reward_experiment_fun(remi_seq)

        pop909_1_rewards = temp_experiment.rewards

        with open(
                os.path.join(project_dir, "phrase-based_generation/ph_bc/data_pkl",
                            "ph_bc-39_sampled_remi_seqs.pkl"), "rb") as f:
            ph_bc_seqs = pickle.load(f)


        temp_experiment = RewardExperimentPhBc(temp_file_name)
        for data in tqdm(ph_bc_seqs):
            remi_seq = data["seq"]
            temp_experiment.reward_experiment_fun(remi_seq)
        
        ph_bc_rewards = temp_experiment.rewards

        t_statistic_melody, p_value_melody = stats.ttest_ind(pop909_1_rewards['melody'], ph_bc_rewards['melody'], equal_var=False)
        t_statistic_chord, p_value_chord = stats.ttest_ind(pop909_1_rewards['chord_progression'], ph_bc_rewards['chord_progression'], equal_var=False)
        print("ph_bc:")
        print(f"Melody rewards: t-statistic = {t_statistic_melody}, p-value = {p_value_melody}")
        print(f"Chord progression rewards: t-statistic = {t_statistic_chord}, p-value = {p_value_chord}")


        with open(
                os.path.join(project_dir, "phrase-based_generation/ph_bc_rl/data_pkl",
                            "ph_bc-39_sampled_remi_seqs.pkl"), "rb") as f:
            ph_bc_rl_seqs = pickle.load(f)

        temp_experiment = RewardExperimentPhBc(temp_file_name)
        for data in tqdm(ph_bc_rl_seqs):
            remi_seq = data["seq"]
            temp_experiment.reward_experiment_fun(remi_seq)

        ph_bc_rl_rewards = temp_experiment.rewards

        t_statistic_melody, p_value_melody = stats.ttest_ind(pop909_1_rewards['melody'], ph_bc_rl_rewards['melody'], equal_var=False)
        t_statistic_chord, p_value_chord = stats.ttest_ind(pop909_1_rewards['chord_progression'], ph_bc_rl_rewards['chord_progression'], equal_var=False)
        print("ph_bc_rl:")
        print(f"Melody rewards: t-statistic = {t_statistic_melody}, p-value = {p_value_melody}")
        print(f"Chord progression rewards: t-statistic = {t_statistic_chord}, p-value = {p_value_chord}")


        with open(
                os.path.join(project_dir, "prompt-based_generation/all_datasets/data_pkl",
                            "REMI-39_sampled_remi_seqs.pkl"), "rb") as f:
            remi_seqs = pickle.load(f)
        
        temp_experiment = RewardExperimentNoPhBc(temp_file_name)
        for data in tqdm(remi_seqs):
            remi_seq = data["seq"]
            temp_experiment.reward_experiment_fun(remi_seq)

        remi_rewards = temp_experiment.rewards

        t_statistic_melody, p_value_melody = stats.ttest_ind(pop909_1_rewards['melody'], remi_rewards['melody'], equal_var=False)
        t_statistic_chord, p_value_chord = stats.ttest_ind(pop909_1_rewards['chord_progression'], remi_rewards['chord_progression'], equal_var=False)
        print("remi:")
        print(f"Melody rewards: t-statistic = {t_statistic_melody}, p-value = {p_value_melody}")
        print(f"Chord progression rewards: t-statistic = {t_statistic_chord}, p-value = {p_value_chord}")


        with open(
                    os.path.join(project_dir, "theme-based_generation/data_pkl",
                                "theme_sampled_remi_seqs.pkl"), "rb") as f:
                theme_seqs = pickle.load(f)

        with open(
                    os.path.join(project_dir, "theme-based_generation/data_pkl",
                                "experiment_input1.pkl"), "rb") as f:
                input1 = pickle.load(f)

        temp_experiment = RewardExperimentNoPhBc(temp_file_name)

        for i in tqdm(range(len(theme_seqs))):
            data = theme_seqs[i]
            input_data = input1[i % len(input1)]
            remi_seq = data["seq"]
            remi_seq = convert_remi_seq(remi_seq)
            remi_seq = fix_ph_bc_label(remi_seq, input_data["phrase_labels"])
            temp_experiment.reward_experiment_fun(remi_seq)
    
        theme_rewards = temp_experiment.rewards
        t_statistic_melody, p_value_melody = stats.ttest_ind(pop909_1_rewards['melody'], theme_rewards['melody'], equal_var=False)
        t_statistic_chord, p_value_chord = stats.ttest_ind(pop909_1_rewards['chord_progression'], theme_rewards['chord_progression'], equal_var=False)
        print("theme:")
        print(f"Melody rewards: t-statistic = {t_statistic_melody}, p-value = {p_value_melody}")
        print(f"Chord progression rewards: t-statistic = {t_statistic_chord}, p-value = {p_value_chord}")