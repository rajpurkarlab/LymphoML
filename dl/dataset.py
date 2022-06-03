from itertools import cycle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import numpy as np
import pandas as pd

import random
import functools

# Classes


class MILDataset(Dataset):
    def __init__(
        self, hdf5_path: str, bag_size=64, is_features: bool = False,
        transform=transforms.ToTensor()
    ):

        self.hdf5_path = hdf5_path
        self.h5data = h5py.File(self.hdf5_path, "r")
        self.core_ids = list(self.h5data.keys())

        self.is_features = is_features
        self.bag_size = bag_size
        self.transform = transform

    def __len__(self):
        return len(self.core_ids)

    def __getitem__(self, idx):

        patient_id = self.core_ids[idx]
        patches: np.ndarray = self.h5data[patient_id][:]

        if len(patches) < self.bag_size:
            patches = [im for im in patches]
        else:
            patches = random.sample([im for im in patches], self.bag_size)

        if self.is_features:
            label = self.h5data[patient_id].attrs["y"]
        else:
            label = self.h5data[patient_id].attrs["label"]
            if self.transform:
                patches = [self.transform(im) for im in patches]

        labels = torch.tensor(int(label))

        return patches, labels



class NaiveDataset(Dataset):
    def __init__(self, hdf5_path: str, is_features: bool = False, transform=transforms.ToTensor(), is_eval: bool = False):

        self.hdf5_path = hdf5_path
        # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        self.h5data = None

        h5data = h5py.File(self.hdf5_path, "r")
        self.core_ids = list(h5data.keys())

        self.lengths = [len(h5data[i]) for i in self.core_ids]
        self.is_features = is_features

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5664, 0.3668, 0.6062),
                                 (0.1906, 0.1613, 0.1336))
        ])
        self.is_eval = is_eval

        # Create mapping df
        mapping_dict = {'cores': [], 'patch_num': []}

        for i in range(len(self.core_ids)):
            mapping_dict['cores'] += self.lengths[i] * [self.core_ids[i]]
            mapping_dict['patch_num'] += list(range(self.lengths[i]))

        self.mapping_df = pd.DataFrame.from_dict(mapping_dict)


    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):

        # Workaround for HDF5 not pickleable: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.h5data is None:
            self.h5data = h5py.File(self.hdf5_path, "r")


        core_idx = 0
        while core_idx < len(self.core_ids):
            if idx - self.lengths[core_idx] < 0:
                break
            idx -= self.lengths[core_idx]
            core_idx += 1

        core_id = self.core_ids[core_idx]
        patch: np.ndarray = self.h5data[core_id][idx]
        if self.is_features:
            label: int = self.h5data[core_id].attrs["y"]
            tensor_patch = torch.tensor(patch)
        else:
            label: int = self.h5data[core_id].attrs["label"]
            if self.transform:
                tensor_patch = self.transform(patch)
            else:
                tensor_patch = torch.tensor(patch)

        if self.is_eval:
            return tensor_patch

        return tensor_patch, torch.tensor(int(label))


if __name__ == "__main__":
    d = NaiveDataset(hdf5_path="/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/data_splits/test.hdf5", is_features=True)
    print(d.mapping_df.head())

