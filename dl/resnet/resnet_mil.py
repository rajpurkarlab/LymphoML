# Imports
import torch
from torch import nn
from torchvision import models

# Local imports
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))  # noqa

from mil_base import MILBase  # noqa

class ResNetMIL(MILBase):
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
            model = models.resnet18(pretrained=True)
        elif self.size == 50:
            model = models.resnet50(pretrained=True)
        else:
            raise NotImplementedError("the size is not supported")

        # if we finetune - only train the classifier, as opposed to e2e - freeze the network
        if self.finetune:
            for param in model.parameters():
                param.requires_grad = False
        
        # set the pretrained weights as the network
        self.feature_extractor = model
        
        # set the linear classifier
        # use the classifier setup in the paper
        self.classifier = nn.Linear(1000, self.hparams.num_classes)

    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x).flatten(1)   # representations
        x = self.classifier(x)                     # classifications
        return x
    

    def aggregate(self, y_hats):
        return torch.max(y_hats, dim=0)[0].unsqueeze(0)
        

if __name__ == "__main__":
    model = ResNetMIL(size=18, lr=1e-5, num_classes=9, finetune=False)
    print(model)

