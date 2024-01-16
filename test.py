import os
import hydra
import lightning.pytorch as pl
import torch

from tricolo.data.data_module import DataModule
from tricolo.model.tricolo_net import TriCoLoNet


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.test_seed, workers=True)
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    model = TriCoLoNet(cfg)

    # load checkpoints
    assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exists."
    ckpt = torch.load(cfg.ckpt_path)["state_dict"]
    to_be_deleted = []
    for key in ckpt:
        if cfg.model.image_encoder is None and "image_encoder" in key:
            to_be_deleted.append(key)
        if cfg.model.voxel_encoder is None and "voxel_encoder" in key:
            to_be_deleted.append(key)
    for key in to_be_deleted:
        del ckpt[key]

    model.load_state_dict(ckpt)
    data_module = DataModule(cfg)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False)
    trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
