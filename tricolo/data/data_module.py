from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from importlib import import_module
import lightning.pytorch as pl
import torch


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(DataModule, self).__init__()
        self.cfg = cfg
        self.dataset = getattr(import_module("tricolo.data.dataset"), cfg.data.dataset)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.cfg, split="train")
            self.val_set = self.dataset(self.cfg, split=self.cfg.inference.split)
        else:
            self.val_set = self.dataset(self.cfg, split=self.cfg.inference.split)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.cfg.data.batch_size, shuffle=True, pin_memory=True,
            num_workers=self.cfg.data.num_workers, drop_last=True, collate_fn=_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.cfg.data.batch_size, pin_memory=True, collate_fn=_collate_fn,
            num_workers=self.cfg.data.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.cfg.data.batch_size, pin_memory=True, collate_fn=_collate_fn,
            num_workers=self.cfg.data.num_workers
        )


def _collate_fn(batch):
    if batch[0]["clip_embeddings_img"] is not None:
        default_collate_items = ("model_id", "category", "tokens", "images", "clip_embeddings_img", "clip_embeddings_text")
    else:
        default_collate_items = ("model_id", "category", "tokens", "images")
    batch_data = []
    voxels_locs_stacked = []
    voxels_feats_stacked = []
    for i, b in enumerate(batch):
        # prepare to collate the default data
        batch_data.append({k: b[k] for k in default_collate_items})
        # prepare to collate special data (voxels_sparse)
        voxels_locs_stacked.append(
            torch.cat((
                torch.full(size=(b["locs"].shape[0], 1), fill_value=i, dtype=torch.int), b["locs"]), dim=1
            )
        )
        voxels_feats_stacked.append(b["feats"])
    # collate the default data
    data_dict = default_collate(batch_data)
    # collate special data (voxels_sparse)
    data_dict["voxels"] = {
        "locs": torch.cat(voxels_locs_stacked),
        "feats": torch.cat(voxels_feats_stacked)
    }
    return data_dict
