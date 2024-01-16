import os
import hydra
import lightning.pytorch as pl
from tricolo.data.data_module import DataModule
from tricolo.model.tricolo_net import TriCoLoNet
from lightning.pytorch.callbacks import LearningRateMonitor
from tricolo.callback.lr_decay_callback import LrDecayCallback


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    lr_decay_callback = LrDecayCallback()
    return [checkpoint_monitor, lr_monitor, lr_decay_callback]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    # hack
    if cfg.model.image_encoder == "CLIPImageEncoder" and cfg.data.image_size != 224:
        print("Error: Please set data.image_size to 224 when using CLIPImageEncoder.")
        exit(0)

    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    os.makedirs(cfg.experiment_output_path, exist_ok=True)

    # load data
    data_module = DataModule(cfg)

    # load model
    model = TriCoLoNet(cfg)

    callbacks = init_callbacks(cfg)

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=hydra.utils.instantiate(cfg.logger))

    ckpt_path = os.path.join(cfg.experiment_output_path, "training",
                             cfg.ckpt_name) if cfg.ckpt_name is not None else None

    if ckpt_path is not None:
        assert os.path.exists(ckpt_path), "Error: Checkpoint path does not exists."
    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
