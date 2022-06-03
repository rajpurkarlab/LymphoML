import torch
from torch import nn
from torchvision import models


# Model adapted from net.py as the template model to load pretrained weights
class TripletNetCore(nn.Module):

    def __init__(self):
        super(TripletNetCore, self).__init__()

        # set the model
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Sequential()
        self.model = model
        self.fc = nn.Sequential(nn.Linear(512*2, 512),
                                 nn.ReLU(True), nn.Linear(512, 256))

    def forward(self, i):

        E1 = self.model(i)
        E2 = self.model(i)
        E3 = self.model(i)

        # Pairwise concatenation of features
        E12 = torch.cat((E1, E2), dim=1)
        E23 = torch.cat((E2, E3), dim=1)
        E13 = torch.cat((E1, E3), dim=1)

        f12 = self.fc(E12)
        f23 = self.fc(E23)
        f13 = self.fc(E13)

        features = torch.cat((f12, f23, f13), dim=1)

        return features