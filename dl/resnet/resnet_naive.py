# Imports
import torch
from torch import nn
from torchvision import models

# Local imports
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))  # noqa

from naive_base import NaiveBase  # noqa

PATH_TO_PRETRAINED = '/deep/group/aihc-bootcamp-fall2021/lymphoma/pretrained/resnet18_he.pt'



class ResNetNaive(NaiveBase):
    def __init__(
        self,
        size: int,
        finetune: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.size = size
        self.finetune = finetune

        if self.size == 18:
            model = models.resnet18()
        elif self.size == 50:
            model = models.resnet50()
        else:
            raise NotImplementedError("Size is not supported")

        if self.finetune:
            for param in model.parameters():
                param.requires_grad = False


        model_dict = model.state_dict()
        weights = {(k, v) for (k, v) in torch.load(PATH_TO_PRETRAINED).items() if k in model_dict}
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        # set the pretrained weights as the network
        self.feature_extractor = model
        self.classifier = nn.Linear(1000, self.hparams.num_classes)


    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = ResNetNaive(size=18, lr=1e-5, num_classes=9, finetune=False)
    print(model)

