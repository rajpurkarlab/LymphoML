# Imports
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

# Local Imports
from naive_base import NaiveBase
from mil_base import MILBase


class LinearNaive(NaiveBase):
    def __init__(self, in_features: int, optimizer, *args, **kwargs):
        super().__init__(optimizer=optimizer, **kwargs)

        print(f'optimizer: {self.optimizer}')
        # set the linear classifier
        self.classifier = nn.Sequential(
                nn.Linear(in_features, self.hparams.num_classes)
                )
        # set the loss criterion -- CE
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        # Assuming `x` is the representation vector
        # Forward step
        x = self.classifier(x)
        return x


class LinearMIL(MILBase):
    def __init__(self, in_features: int, *args, **kwargs):
        super().__init__(**kwargs)

        # set the linear classifier
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)

        # set the loss criterion -- CE
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Assuming `x` is the representation vector

        # Forward step
        x = self.classifier(x)
        return x

    def aggregate(self, y_hats):
        return torch.max(y_hats, dim=0)[0].unsqueeze(0)


if __name__ == "__main__":
    # model_naive = LinearNaive(256 * 3, lr=1e-5, num_classes=9, fine_tune=False)
    # model_mil = LinearMIL(256 * 3, lr=1e-5, num_classes=9, fine_tune=False)
    # print(model_naive, model_mil)

    l = LinearNaive(256*3, lr=1e-2, num_classes=8)
    print(l)
