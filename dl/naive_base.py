import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import ConfusionMatrix

from collections import OrderedDict

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

class NaiveBase(pl.LightningModule):
    def __init__(self, num_classes: int = 9, weights: torch.Tensor = None, optimizer='Adam', lr=1e-3):
        super().__init__()

        self.num_classes = num_classes

        self.optimizer = optimizer

        self.lr = lr

        # Ensure variables are accessible via `hparams` attribute
        self.save_hyperparameters()

        # # Weighted crossentropy for dataset skew
        # (https://github.com/AdeelH/pytorch-multi-class-focal-loss)
        self.criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.tensor([0.05, 0.05, 0.125, 0.1, 0.1, 0.125, 0.1, 0.1, 0.25]),
            gamma=2,
            reduction='mean',
            force_reload=False
        )


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
        # Forward step
        raise NotImplementedError("forward must be implemented for inheriting NaiveBase")

    def configure_optimizers(self):
        # only train parameters that are not frozen
        parameters = self.parameters()
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))

        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr)
            print(f'learning rate is: {self.lr}')
        else:
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.lr, momentum=0.9)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.train_acc(y_hat, y)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.val_acc(y_hat, y)

        # log validation metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": y_hat.detach(), "targets": y.detach()}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure("train_cm", fig_, self.current_epoch)

    def validation_epoch_end(self, outputs):
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        confusion_matrix = self.val_cm(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("val_cm", fig_, self.current_epoch)

    def test_epoch_end(self, outputs):

        targets = torch.cat([tmp['targets'] for tmp in outputs])
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        confusion_matrix = self.test_cm(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(self.num_classes), columns=range(self.num_classes))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("test_cm", fig_, self.current_epoch)

