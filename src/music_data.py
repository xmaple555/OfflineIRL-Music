import torch
from torch.utils.data.dataset import Dataset
import pickle
from copy import deepcopy


class MusicDataset(Dataset):

    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.constants = {
            "max_tgt_len": max([len(x["tgt"]) for x in self.data])
        }
        print(self.constants)

    def __getitem__(self, index):
        tgt = deepcopy(self.data[index]["tgt"])

        # padding
        tgt_msk = []
        if len(tgt) > self.constants["max_tgt_len"]:
            print("Should not be here")
            assert True

        else:
            tgt_msk = [0] * len(tgt) + [1] * (self.constants["max_tgt_len"] -
                                              len(tgt))
            tgt.extend([0] * (self.constants["max_tgt_len"] - len(tgt)))

        current_entry = {
            "tgt": tgt,
            "tgt_msk": tgt_msk,
        }

        assert (len(tgt) == self.constants["max_tgt_len"])
        assert (len(tgt_msk) == self.constants["max_tgt_len"])

        return {key: torch.tensor(value) for key, value in current_entry.items()}

    def __len__(self):
        return len(self.data)


class RLMusicDataset(Dataset):

    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.constants = {
            "max_tgt_len": max([len(x["tgt"]) for x in self.data])
        }
        print(self.constants)

    def __getitem__(self, index):
        tgt = deepcopy(self.data[index]["tgt"])

        # padding
        tgt_msk = []
        if len(tgt) > self.constants["max_tgt_len"]:
            print("Should not be here")
            assert True

        else:
            tgt_msk = [0] * len(tgt) + [1] * (self.constants["max_tgt_len"] -
                                              len(tgt))
            tgt.extend([0] * (self.constants["max_tgt_len"] - len(tgt)))

        current_entry = {
            "tgt": tgt,
            "tgt_msk": tgt_msk,
            "reward": float(self.data[index]["reward"]),
        }

        assert (len(tgt) == self.constants["max_tgt_len"])
        assert (len(tgt_msk) == self.constants["max_tgt_len"])

        return {key: torch.tensor(value) for key, value in current_entry.items()}

    def __len__(self):
        return len(self.data)


class ChordDataset(Dataset):

    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.constants = {"max_tgt_len": max([len(x) for x in self.data])}
        print(self.constants)

    def __getitem__(self, index):
        tgt = deepcopy(self.data[index])

        # padding
        tgt_msk = []
        if len(tgt) > self.constants["max_tgt_len"]:
            print("Should not be here")
            assert True

        else:
            tgt_msk = [0] * len(tgt) + [1] * (self.constants["max_tgt_len"] -
                                              len(tgt))
            tgt.extend([0] * (self.constants["max_tgt_len"] - len(tgt)))

        current_entry = {
            "tgt": tgt,
            "tgt_msk": tgt_msk,
        }

        assert (len(tgt) == self.constants["max_tgt_len"])
        assert (len(tgt_msk) == self.constants["max_tgt_len"])

        return {key: torch.tensor(value) for key, value in current_entry.items()}

    def __len__(self):
        return len(self.data)


class MelodyDataset(Dataset):

    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.constants = {"max_tgt_len": max([len(x) for x in self.data])}
        print(self.constants)

    def __getitem__(self, index):
        tgt = deepcopy(self.data[index])

        # padding
        tgt_msk = []
        if len(tgt) > self.constants["max_tgt_len"]:
            print("Should not be here")
            assert True

        else:
            tgt_msk = [0] * len(tgt) + [1] * (self.constants["max_tgt_len"] -
                                              len(tgt))
            tgt.extend([0] * (self.constants["max_tgt_len"] - len(tgt)))

        current_entry = {
            "tgt": tgt,
            "tgt_msk": tgt_msk,
        }

        assert (len(tgt) == self.constants["max_tgt_len"])
        assert (len(tgt_msk) == self.constants["max_tgt_len"])

        return {key: torch.tensor(value) for key, value in current_entry.items()}

    def __len__(self):
        return len(self.data)
