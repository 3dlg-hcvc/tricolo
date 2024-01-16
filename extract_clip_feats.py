import os
import clip
import hydra
import lightning.pytorch as pl
from importlib import import_module
from torch.utils.data import DataLoader
from tricolo.data.data_module import _collate_fn
from tqdm import tqdm
import torch


@torch.no_grad()
def run_epoch(cfg, clip_model, dataloader, split):
    model_ids = []
    avg_img_embeddings = []
    text_embeddings = []
    for i, data_dict in enumerate(tqdm(dataloader)):
        # images
        img_output = clip_model.encode_image(
            data_dict["images"].flatten(end_dim=1).to("cuda")
        ).reshape(-1, cfg.data.num_views, clip_model.visual.output_dim)

        img_output = torch.mean(img_output, dim=1)
        img_output /= img_output.norm(dim=1, keepdim=True)

        model_ids += data_dict["model_id"]
        avg_img_embeddings.append(img_output.cpu())

        # texts
        text_output = clip_model.encode_text(data_dict["tokens"].to("cuda"))
        text_output /= text_output.norm(dim=1, keepdim=True)

        text_embeddings.append(text_output.cpu())

    avg_img_embeddings = torch.cat(avg_img_embeddings)
    text_embeddings = torch.cat(text_embeddings)

    output_data = {}
    for i, model_id in enumerate(model_ids):
        output_data[model_id] = {"img": avg_img_embeddings[i], "text": text_embeddings[i]}

    save_path = os.path.join(cfg.data.exp_data_root_path, f"clip_embeddings_{split}.pth")
    # save clip embeddings
    torch.save(output_data, save_path)
    print(f"Pre-trained CLIP embeddings are saved at {save_path}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # hack
    cfg.model.text_encoder = "CLIPTextEncoder"

    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # load clip
    clip_model = clip.load(cfg.model.modules.clip_model, device="cuda")[0]

    # freeze CLIP
    for param in clip_model.parameters():
        param.requires_grad = False

    # load data
    dataset = getattr(import_module("tricolo.data.dataset"), cfg.data.dataset)

    for split in ("train", "val", "test"):
        split_dataset = dataset(cfg, split=split)
        loader = DataLoader(
            split_dataset, batch_size=cfg.data.batch_size, shuffle=False, pin_memory=True,
            num_workers=cfg.data.num_workers, collate_fn=_collate_fn
        )
        run_epoch(cfg, clip_model, loader, split)


if __name__ == '__main__':
    main()
