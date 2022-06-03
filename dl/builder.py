import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

# Local imports

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "./tripletnet"))
sys.path.append(os.path.join(os.getcwd(), "./resnet"))

from dataset import NaiveDataset, MILDataset
from linear import LinearNaive, LinearMIL
from tripletnet.tripletnet_naive import TripletNetNaive
from resnet.resnet_naive import ResNetNaive
from tripletnet.tripletnet_mil import TripletNetMIL
from resnet.resnet_mil import ResNetMIL

# Lightning seed

pl.seed_everything(42)

# Constants

import json
CONFIG = json.load(open("../config/config.json", "r"))

RAW = lambda group: os.path.join(CONFIG["data_splits"], f"{group}.hdf5")
FEATURES = lambda model, group: os.path.join(CONFIG["data_splits"], f"custom_splits/{model}_features/{model}_{group}_features.hdf5")
AUG =  lambda group: os.path.join(CONFIG["data_splits"], f"augmented/{group}.hdf5")
CKPT_DIR = lambda model: os.path.join(CONFIG["checkpoints"], f"{model}")
PREDS_PATH = lambda model: os.path.join(CONFIG["predictions"] ,f"dl/{model}.csv")

# Calculated for our particular dataset
CORE_PROPORTIONS = [0.4719, 0.1770, 0.0148, 0.0771, 0.0948, 0.0277, 0.0807, 0.0508, 0.0051]

# Helper Functions

def get_weights():
    weights = 1. / torch.tensor(CORE_PROPORTIONS)
    weights = weights / sum(weights)
    return weights

def make_naive_model(
    model_name: str, use_stored_features: bool,
    lr: float, num_classes: int,
    finetune: bool,
    optimizer: str
):
    """Make model for Naive training"""

    weights = get_weights()
    return {
        'tripletnet': LinearNaive(256 * 3, lr=lr, num_classes=num_classes,
        weights=weights, optimizer=optimizer) if use_stored_features
        else TripletNetNaive(
            finetune=finetune, lr=lr, num_classes=num_classes, weights=weights),
        'tripletnet_e2e': TripletNetNaive(
            finetune=finetune,
            lr=lr, num_classes=num_classes,
            weights=weights, optimizer=optimizer
        ),
        'resnet': LinearNaive(512, lr=lr, num_classes=num_classes,
        weights=weights, optimizer=optimizer) if use_stored_features
        else ResNetNaive(
            finetune=finetune, lr=lr, num_classes=num_classes, weights=weights),
        'resnet_e2e': ResNetNaive(
            size=18,
            finetune=finetune,
            lr=lr, num_classes=num_classes,
            weights=weights, optimizer=optimizer
        )
    }[model_name]


def make_naive_dataloaders(
    num_workers: int, 
    batch_size: int, 
    use_stored_features: bool = False, 
    aug_data: bool = True, 
    model: str = None
):
    """Create dataloaders for Naive training"""
    if use_stored_features:
        paths = {'train': FEATURES(model, 'train'), 'val': FEATURES(model, 'val'), 'test': FEATURES(model, 'test')}
        datasets = {i: NaiveDataset(hdf5_path=paths[i], is_features=True) for i in paths}
    elif aug_data:
        paths = {'train': AUG('train'), 'val': AUG('val'), 'test': AUG('test')}
        datasets = {i: NaiveDataset(hdf5_path=paths[i], is_features=False) for i in paths}
    else:
        paths = {'train': RAW('train'), 'val': RAW('val'), 'test': RAW('test')}
        datasets = {i: NaiveDataset(hdf5_path=paths[i], is_features=False) for i in paths}

    dataloaders = {
        i: DataLoader(datasets[i], batch_size=batch_size, pin_memory=True,  num_workers=num_workers, shuffle=(i=="train"))
        for i in datasets
    }

    return dataloaders

def make_inference_dataloader(
    num_workers: int,
    batch_size: int,
    use_stored_features: bool = False,
    aug_data: bool = True,
    model: str = None
):

    # If finetuning
    if use_stored_features:
        dataset = NaiveDataset(hdf5_path=FEATURES(model, 'test'), is_features=True, is_eval=True)
    elif aug_data:
        dataset = NaiveDataset(hdf5_path=AUG('test'), is_features=False, is_eval=True)
    else:
        dataset = NaiveDataset(hdf5_path=RAW('test'), is_features=False, is_eval=True)

    return DataLoader(dataset, batch_size=batch_size, pin_memory=True,  num_workers=num_workers), dataset.mapping_df


def make_mil_model(model_name: str, use_stored_features: bool):
    """ Make model for MIL training"""
    weights = get_weights()

    return {
        'tripletnet': \
            LinearMIL(256 * 3, lr=1e-3, num_classes=9, weights=weights) if use_stored_features
            else TripletNetMIL(finetune=False ,lr=1e-3, num_classes=9, weights=weights),
        'tripletnet_nonDLBCL': \
            LinearMIL(256 * 3, lr=1e-3, num_classes=8, weights=weights) if use_stored_features
            else TripletNetMIL(finetune=False, lr=1e-3, num_classes=8, weights=weights),
        'tripletnet_e2e': TripletNetMIL(finetune=True, lr=1e-3, num_classes=9, weights=weights),
        'resnet': ResNetMIL(size=18, lr=1e-3, num_classes=9, finetune=True, weights=weights),
        'resnet_e2e': ResNetMIL(size=18, lr=1e-3, num_classes=9, finetune=False, weights=weights),
    }[model_name]


def make_mil_dataloaders(num_workers: int, batch_size: int, use_stored_features: bool = False, model: str = None):
    """Create dataloaders for MIL training"""
    # If finetuning
    if use_stored_features:
        paths = {'train': FEATURES(model, 'train'), 'val': FEATURES(model, 'val'), 'test': FEATURES(model, 'test')}
        datasets = {i: MILDataset(hdf5_path=paths[i], is_features=True) for i in paths}
    else:
        paths = {'train': RAW('train'), 'val': RAW('val'), 'test': RAW('test')}
        datasets = {i: MILDataset(hdf5_path=paths[i], is_features=False) for i in paths}

    dataloaders = {
        i: DataLoader(datasets[i], batch_size=batch_size, num_workers=num_workers, shuffle=True)
        for i in datasets
    }

    return dataloaders
