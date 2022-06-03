# Imports
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import ConfusionMatrix

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Main Class

class MILBase(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, num_classes: int = 8, weights: torch.Tensor = None):
        super().__init__()

        self.num_classes = num_classes
        # Ensure variables are accessible via `hparams` attribute
        self.save_hyperparameters()

        # # Weighted crossentropy for dataset skew
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        # set the loss criterion -- Focal Loss 
        # (https://github.com/AdeelH/pytorch-multi-class-focal-loss)
        # self.criterion = torch.hub.load(
        #     'adeelh/pytorch-multi-class-focal-loss',
        #     model='FocalLoss',
        #     alpha=torch.tensor([0.05, 0.05, 0.125, 0.1, 0.1, 0.125, 0.1, 0.1, 0.25]),
        #     gamma=2,
        #     reduction='mean',
        #     force_reload=False
        # )


        # Classification metrics

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        acc = Accuracy()
        self.train_acc = acc.clone()
        self.val_acc = acc.clone()
        self.test_acc = acc.clone()

        # Add confusion matrix into training metrics
        cm = ConfusionMatrix(self.hparams.num_classes)
        self.train_cm = cm.clone()
        self.val_cm = cm.clone()
        self.test_cm = cm.clone()

    def forward(self, x):
        raise NotImplementedError("Forward must be implemented for inheriting MILBase")

    def aggregate(self, y_hats):
        raise NotImplementedError("Aggregate must be implemented for inheriting MILBase")

    def bag_forward(self, bag):
        y_hats = []
        for x in bag:
            y_hats.append(self(x).squeeze())

        y_hat = self.aggregate(torch.stack(y_hats, dim=0))

        return y_hat

    def configure_optimizers(self):
        # only train parameters that are not frozen
        parameters = self.parameters()
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))

        optimizer = torch.optim.Adam(trainable_parameters, lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.bag_forward(bag)
        loss = self.criterion(y_hat, y)
        acc = self.train_acc(y_hat, y)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}

    def validation_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.bag_forward(bag)
        loss = self.criterion(y_hat, y)
        acc = self.val_acc(y_hat, y)

        # log validation metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}

    def test_step(self, batch, batch_idx):
        bag, y = batch
        y_hat = self.bag_forward(bag)
        loss = self.criterion(y_hat, y)
        acc = self.test_acc(y_hat, y)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}

    def training_epoch_end(self, outputs):

        targets = torch.cat([tmp['targets'] for tmp in outputs])
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        confusion_matrix = self.train_cm(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure("train_cm", fig_, self.current_epoch)

    def validation_epoch_end(self, outputs):

        targets = torch.cat([tmp['targets'] for tmp in outputs])
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        confusion_matrix = self.val_cm(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("val_cm", fig_, self.current_epoch)

    def test_epoch_end(self, outputs):

        targets = torch.cat([tmp['targets'] for tmp in outputs])
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        confusion_matrix = self.test_cm(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("test_cm", fig_, self.current_epoch)
