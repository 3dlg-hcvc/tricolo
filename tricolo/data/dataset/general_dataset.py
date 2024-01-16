from torchvision.transforms import Resize, InterpolationMode, Normalize
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import json
import clip
import os


class GeneralDataset(Dataset):
    def __init__(self, cfg, split):
        json_file_path = getattr(cfg.data, f"{split}_lang_data_path")
        self.voxel_size = cfg.data.voxel_size
        self.language_data = []
        self.cfg = cfg
        self.split = split
        self.use_clip_tokens = cfg.model.text_encoder == "CLIPTextEncoder"
        with open(json_file_path, "r") as f:
            raw_language_data = json.load(f)

        # TODO
        clip_embeddings = None
        clip_embeddings_path = os.path.join(cfg.data.exp_data_root_path, f"clip_embeddings_{split}.pth")
        if (os.path.exists(clip_embeddings_path) and
                (cfg.model.text_encoder == "CLIPTextEncoder" or cfg.model.image_encoder == "CLIPImageEncoder")):
            # use pre-trained CLIP embeddings
            clip_embeddings = torch.load(clip_embeddings_path)
        self.vision_data_dict = {}
        for language_data_item in tqdm(raw_language_data, desc=f"Loading {split} data from disk"):
            self.language_data.append({
                "model_id": language_data_item["model_id"],
                "category": language_data_item["category"],
                "tokens": np.asarray(language_data_item["tokens"], dtype=np.int16),
                "text": language_data_item["caption"].strip()
            })
            if (language_data_item["category"], language_data_item["model_id"]) not in self.vision_data_dict:
                npz_data = np.load(
                    os.path.join(
                        cfg.data.exp_data_root_path, language_data_item["category"],
                        f"{language_data_item['model_id']}.npz"
                    )
                )
                dense_voxel_data = npz_data[f"voxel{cfg.data.voxel_size}"]

                # convert to sparse format
                grid = np.transpose(dense_voxel_data, (1, 2, 3, 0))
                grid_flatten = grid.reshape(-1, grid.shape[3])
                solid_mask = grid_flatten[:, -1].nonzero()
                coords = (np.indices(grid.shape[:3], dtype=np.uint8).reshape(3, -1).T)[solid_mask]
                feats = grid_flatten[:, :3][solid_mask]

                sub_idx = np.round(np.linspace(0, len(npz_data["images"]) - 1, cfg.data.num_views)).astype(int)
                multi_view_imgs = npz_data["images"][sub_idx]

                if clip_embeddings is not None:
                    clip_embeddings_img = clip_embeddings[language_data_item["model_id"]]["img"].to(torch.float32)
                    clip_embeddings_text = clip_embeddings[language_data_item["model_id"]]["text"].to(torch.float32)
                else:
                    clip_embeddings_img = None
                    clip_embeddings_text = None
                self.vision_data_dict[(language_data_item["category"], language_data_item["model_id"])] = {
                    "images": Resize(
                        cfg.data.image_size, interpolation=InterpolationMode.BICUBIC, antialias=True
                    )(torch.from_numpy(multi_view_imgs)),
                    "voxels": (coords, feats),
                    "clip_embeddings_img": clip_embeddings_img,
                    "clip_embeddings_text": clip_embeddings_text
                }

    def __len__(self):
        return len(self.language_data)

    def __getitem__(self, idx):
        language_data_item = self.language_data[idx]
        if self.use_clip_tokens:
            tokens = clip.tokenize(language_data_item["text"], truncate=True)[0]
        else:
            tokens = language_data_item["tokens"].astype(np.int32)

        data_dict = {
            "model_id": language_data_item["model_id"],
            "category": language_data_item["category"],
            "tokens": tokens
        }
        vision_data = self.vision_data_dict[(self.language_data[idx]['category'], self.language_data[idx]['model_id'])]
        data_dict["images"] = Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )(vision_data["images"].to(torch.float32) / 255)
        

        data_dict["locs"] = torch.from_numpy(vision_data["voxels"][0].astype(np.int32))
        data_dict["feats"] = torch.from_numpy(vision_data["voxels"][1].astype(np.float32) / 255)

        data_dict["clip_embeddings_img"] = vision_data["clip_embeddings_img"]
        data_dict["clip_embeddings_text"] = vision_data["clip_embeddings_text"]

        return data_dict

