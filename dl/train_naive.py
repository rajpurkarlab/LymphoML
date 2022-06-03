import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from builder import *

def train(cfg):

    # load model
    model = make_naive_model(cfg.model_type, cfg.stored_features,
                       cfg.lr, cfg.num_classes, cfg.finetune, cfg.layers_tune,
                       cfg.optimizer)

    # load dataloaders
    dataloaders = make_mil_dataloaders(
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        use_stored_features=cfg.stored_features,
        aug_data=cfg.aug_data,
        model=cfg.model_type
    )

    # get checkpoint directory to save model to depending on model type: resnet/tripletnet/etc.
    ckpt_dir = CKPT_DIR(cfg.model_type)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    print("Checkpoints:", ckpt_dir)

    # training
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_dir, cfg.ckpt_name),
        monitor="val/acc"
    )

    trainer = pl.Trainer(
        gpus=2, num_nodes=1,
        max_epochs=cfg.epochs, log_every_n_steps=1,
        default_root_dir=os.path.join(ckpt_dir),
        callbacks=[checkpoint_callback],
        accelerator="ddp"
    )

    trainer.fit(model, dataloaders['train'], dataloaders['val'])
    trainer.test(model, dataloaders['test'])

    # Save checkpoint and state dict
    trainer.save_checkpoint(os.path.join(ckpt_dir, f"end_{cfg.ckpt_name}.ckpt"))
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"end_{cfg.ckpt_name}.pt"))


@hydra.main(config_path="configs", config_name='config')
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
