import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core.mixins import device_dtype_mixin

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Local imports

from builder import *

def train(cfg):

    model = make_mil_model(cfg.model_type, cfg.stored_features)

    dataloaders = make_mil_dataloaders(
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        use_stored_features=cfg.stored_features,
        model=cfg.model_type
    )

    # training
    trainer = pl.Trainer(
        gpus=[0], num_nodes=1,
        precision=16, limit_train_batches=0.5,
        max_epochs=cfg.epochs, log_every_n_steps=1,
        accelerator="ddp"
    )
    trainer.fit(model, dataloaders['train'], dataloaders['val'])

    trainer.test(model, dataloaders['test'])

    trainer.save_checkpoint(f"../../models/{cfg.ckpt_name}.ckpt")


@hydra.main(config_path="configs", config_name="mil")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
