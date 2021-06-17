import pickle

import torch
from torch.utils.data import Dataset


class BertCommonsenseData(Dataset):

    def __init__(self, data_path):

        def load_data():
            with open(data_path, 'rb') as input:
                return pickle.load(input)

        def process_samples(samples):
            p_samples = []
            for sample in samples:
                p_sample = {}
                for k, v in sample.items():
                    if "net" in k:
                        arr = k.split(".")
                        if arr[0] not in p_sample:
                            p_sample[arr[0]] = {}
                        if "len" in arr[1]:
                            p_sample[arr[0]][arr[1]] = torch.tensor(v)
                        else:
                            p_sample[arr[0]][arr[1]] = v
                    else:
                        if type(v) is not list:
                            v = torch.tensor(v)
                        p_sample[k] = v
                p_samples.append(p_sample)
            return p_samples
        # TODO: memory problems with the launcher.py
        self.data = process_samples(load_data())[:100]
        self.start = 0
        self.end = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
