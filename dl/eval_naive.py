# Imports
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from itertools import chain
import os

from builder import *

# Lightning seed

pl.seed_everything(42)

# Main function
def test(cfg):

    model = make_naive_model(cfg.model_type, cfg.stored_features,
                       cfg.lr, cfg.num_classes, cfg.finetune, cfg.layers_tune,
                       cfg.optimizer)

    dataloader, mapping_df = make_inference_dataloader( num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        use_stored_features=cfg.stored_features,
        aug_data=cfg.aug_data,
        model=cfg.model_type
    )

    trainer = pl.Trainer(
        gpus=[0], num_nodes=1,
        precision=16, limit_train_batches=0.5,
        max_epochs=cfg.epochs, log_every_n_steps=1,
        accelerator="ddp"
    )

    ckpt_dir = CKPT_DIR(cfg.model_type)
    model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, f"end_{cfg.ckpt_name}.pt"))["state_dict"]
    )

    batch_logits = trainer.predict(model, dataloaders=dataloader, return_predictions=True)

    preds = list(chain(*[torch.argmax(logits, dim=1).tolist() for logits in batch_logits]))

    preds_df = mapping_df.copy()
    preds_df['pred'] = preds

    preds_path = PREDS_PATH(cfg.model_type)
    preds_df.to_csv(preds_path)

@hydra.main(config_path="configs", config_name='config')
def main(cfg: DictConfig) -> None:
    test(cfg)


if __name__ == "__main__":
    main()
