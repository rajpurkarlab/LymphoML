# Imports

import torch
import h5py
from tqdm import tqdm
from pprint import pprint

# Local imports

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "../"))

from dataset import NaiveDataset

# Paths
# Read Config
config_path = '../config/config.json'
config = json.load(open("../config/config.json", "r"))

PATH_TO_TRAIN = os.path.join(config["data_splits"], "train.hdf5")
PATH_TO_VAL = os.path.join(config["data_splits"], "val.hdf5")
PATH_TO_TEST = os.path.join(config["data_splits"], "test.hdf5")


# Helper functions

def get_weights(hdf5_path: str):
    h5data = h5py.File(hdf5_path, "r")
    cores = list(h5data.keys())
    
    patch_weights = [0 for _ in range(9)]
    core_weights = [0 for _ in range(9)]
    
    for core in tqdm(cores):
        label = h5data[core].attrs["label"]
        patch_weights[label] += len(h5data[core])
        core_weights[label] += 1
    
    patch_weights = torch.tensor(patch_weights) / sum(patch_weights)
    core_weights = torch.tensor(core_weights) / sum(core_weights)
    
    return {'patch': patch_weights, 'core': core_weights}
          
def main():
    paths = {'train': PATH_TO_TRAIN, 'val': PATH_TO_VAL, 'test': PATH_TO_TEST}
    weights = {i: get_weights(paths[i]) for i in paths}
    
    pprint(weights)
          
          
if __name__ == "__main__":
    main()
