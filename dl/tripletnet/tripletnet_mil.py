# Imports
import torch
from torch import nn

from collections import OrderedDict

# Local imports
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))  # noqa

from mil_base import MILBase  # noqa
from tripletnet_core import TripletNetCore # noqa

# For `tripletnet_core.py` to be visible for higher dirs

sys.path.append(os.getcwd()) # noqa

# Path to pre-trained TripleNet model
PATH_TO_PRETRAINED = '/deep/group/aihc-bootcamp-fall2021/lymphoma/models/Camelyon16_pretrained_model.pt'


class TripletNetMIL(MILBase):
    def __init__(self, finetune: bool = False, *args, **kwargs):
        super().__init__(**kwargs)
                
        self.finetune = finetune

        # Load pre-trained network:
        state_dict = torch.load(PATH_TO_PRETRAINED) ## TODO: change this to pytorch lightning

        # create new OrderedDict that does not contain `module`
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
            
        # load pretrained weights onto TripletNet model
        model = TripletNetCore()
        model.load_state_dict(new_state_dict)
        
        # if we finetune - only train the classifier, as opposed to e2e - freeze the network
        if self.finetune:
            for name, param in enumerate(model.named_parameters()):
                param = param[1]
                param.requires_grad=False
        
        # set the pretrained weights as the network
        self.feature_extractor = model
        
        # set the linear classifier
        # use the classifier setup in the paper
        self.classifier = nn.Linear(256*3, self.hparams.num_classes)
    

    def forward(self, x):
        # Forward step
        x = self.feature_extractor(x).flatten(1)   # representations
        x = self.classifier(x)                     # classifications
        return x

    def aggregate(self, y_hats):
        return torch.max(y_hats, dim=0)[0].unsqueeze(0)
        

if __name__ == "__main__":
    model = TripletNetMIL(finetune=False, lr=1e-5, num_classes=9)
    print(model)

