from reward import *
import numpy as np
from vocab import Vocab
from model import ChordModel, MelodyModel

myvocab = Vocab()

import os

project_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
))


class RewardExperimentPhBc:

    def __init__(self, file_name) -> None:
        self.rewards = {
            "melody": [],
            "chord_progression": [],
        }

        self.total_number = 0
        self.file_name = file_name
        self.chord_model = ChordModel(myvocab.n_tokens).cuda()

        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward", "ph_bc", "ckpt",
                         "chord_model", "chord_model_20.pt"))

        try:
            self.chord_model.load_state_dict(checkpoint['model_state_dict'])
        except:
            self.chord_model = ChordModel(829).cuda()
            self.chord_model.load_state_dict(checkpoint['model_state_dict'])

        self.melody_model = MelodyModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward", "ph_bc", "ckpt",
                         "melody_model", "melody_model_35.pt"))
        self.melody_model.load_state_dict(checkpoint['model_state_dict'])

    def reward_experiment_fun(self, remi_seq):
        self.chord_model.eval()
        self.melody_model.eval()

        melody_reward_value = melody_reward(myvocab,
                                            remi_seq,
                                            self.melody_model,
                                            remove_ph_bc=False)

        chord_progression_reward_value = chord_progression_reward(
            myvocab, self.chord_model, remi_seq, remove_ph_bc=False)

        self.rewards["melody"].append(melody_reward_value)
        self.rewards["chord_progression"].append(chord_progression_reward_value)

        with open(self.file_name, 'w') as f:
            self.total_number += 1
            f.write(f"total_number: {self.total_number}\n\n")
            f.write("reward\n\n")
            for key, value in self.rewards.items():
                f.write(
                    f"{key}: mean = {np.mean(value)}, std = {np.std(value)}\n")


class RewardExperimentNoPhBc:

    def __init__(self, file_name) -> None:
        self.rewards = {
            "melody": [],
            "chord_progression": [],
        }

        self.total_number = 0
        self.file_name = file_name
        self.chord_model = ChordModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward", "ph_bc", "ckpt",
                         "chord_model", "chord_model_20.pt"))

        try:
            self.chord_model.load_state_dict(checkpoint['model_state_dict'])
        except:
            self.chord_model = ChordModel(829).cuda()
            self.chord_model.load_state_dict(checkpoint['model_state_dict'])

        self.melody_model = MelodyModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward", "ph_bc", "ckpt",
                         "melody_model", "melody_model_35.pt"))
        self.melody_model.load_state_dict(checkpoint['model_state_dict'])

    def reward_experiment_fun(self, remi_seq):
        self.chord_model.eval()
        self.melody_model.eval()

        melody_reward_value = melody_reward(myvocab,
                                            remi_seq,
                                            self.melody_model,
                                            remove_ph_bc=False)

        chord_progression_reward_value = chord_progression_reward(
            myvocab, self.chord_model, remi_seq, remove_ph_bc=False)

        self.rewards["melody"].append(melody_reward_value)
        self.rewards["chord_progression"].append(chord_progression_reward_value)

        with open(self.file_name, 'w') as f:
            self.total_number += 1
            f.write(f"total_number: {self.total_number}\n\n")
            f.write("reward\n\n")
            for key, value in self.rewards.items():
                f.write(
                    f"{key}: mean = {np.mean(value)}, std = {np.std(value)}\n")


class RewardExperimentNoPhBc2:

    def __init__(self, file_name) -> None:
        self.rewards = {
            "melody": [],
            "chord_progression": [],
        }

        self.total_number = 0
        self.file_name = file_name
        self.chord_model = ChordModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward", "no_ph_bc", "ckpt",
                         "chord_model", "chord_model_20.pt"))
        self.chord_model.load_state_dict(checkpoint['model_state_dict'])

        self.melody_model = MelodyModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward", "no_ph_bc", "ckpt",
                         "melody_model", "melody_model_35.pt"))
        self.melody_model.load_state_dict(checkpoint['model_state_dict'])

    def reward_experiment_fun(self, remi_seq):
        self.chord_model.eval()
        self.melody_model.eval()

        melody_reward_value = melody_reward(myvocab,
                                            remi_seq,
                                            self.melody_model,
                                            remove_ph_bc=True)

        chord_progression_reward_value = chord_progression_reward(
            myvocab, self.chord_model, remi_seq, remove_ph_bc=True)

        self.rewards["melody"].append(melody_reward_value)
        self.rewards["chord_progression"].append(chord_progression_reward_value)

        with open(self.file_name, 'w') as f:
            self.total_number += 1
            f.write(f"total_number: {self.total_number}\n\n")
            f.write("reward\n\n")
            for key, value in self.rewards.items():
                f.write(
                    f"{key}: mean = {np.mean(value)}, std = {np.std(value)}\n")


class RewardExperimentPhBc2:

    def __init__(self, file_name) -> None:
        self.rewards = {
            "melody": [],
            "chord_progression": [],
        }

        self.total_number = 0
        self.file_name = file_name
        self.chord_model = ChordModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward",
                         "ph_bc_only_pop909", "ckpt", "chord_model",
                         "chord_model_20.pt"))
        self.chord_model.load_state_dict(checkpoint['model_state_dict'])

        self.melody_model = MelodyModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward",
                         "ph_bc_only_pop909", "ckpt", "melody_model",
                         "melody_model_35.pt"))
        self.melody_model.load_state_dict(checkpoint['model_state_dict'])

    def reward_experiment_fun(self, remi_seq):
        self.chord_model.eval()
        self.melody_model.eval()

        melody_reward_value = melody_reward(myvocab,
                                            remi_seq,
                                            self.melody_model,
                                            remove_ph_bc=False)

        chord_progression_reward_value = chord_progression_reward(
            myvocab, self.chord_model, remi_seq, remove_ph_bc=False)

        self.rewards["melody"].append(melody_reward_value)
        self.rewards["chord_progression"].append(chord_progression_reward_value)

        with open(self.file_name, 'w') as f:
            self.total_number += 1
            f.write(f"total_number: {self.total_number}\n\n")
            f.write("reward\n\n")
            for key, value in self.rewards.items():
                f.write(
                    f"{key}: mean = {np.mean(value)}, std = {np.std(value)}\n")


class MelodyRepetitionExperimentPhBc:

    def __init__(self, file_name) -> None:
        self.rewards = {
            "melody": [],
            "melody_repetition": [],
        }

        self.total_number = 0
        self.file_name = file_name

        self.melody_model = MelodyModel(myvocab.n_tokens).cuda()
        checkpoint = torch.load(
            os.path.join(project_dir, "melody_chord_reward", "ph_bc", "ckpt",
                         "melody_model", "melody_model_35.pt"))
        self.melody_model.load_state_dict(checkpoint['model_state_dict'])

    def reward_experiment_fun(self, remi_seq):
        self.melody_model.eval()

        melody_reward_value = melody_reward(myvocab,
                                            remi_seq,
                                            self.melody_model,
                                            remove_ph_bc=False)

        melody_repetition_value = melody_repetition(
            myvocab,
            remi_seq,
        )

        self.rewards["melody"].append(melody_reward_value)
        self.rewards["melody_repetition"].append(melody_repetition_value)

        with open(self.file_name, 'w') as f:
            self.total_number += 1
            f.write(f"total_number: {self.total_number}\n\n")
            f.write("reward\n\n")
            for key, value in self.rewards.items():
                f.write(
                    f"{key}: mean = {np.mean(value)}, std = {np.std(value)}\n")

