import torch
import torch.nn as nn
import lightning.pytorch as pl


class CLIPImageEncoder(pl.LightningModule):
    def __init__(self, clip_model, out_dim, num_views):
        super(CLIPImageEncoder, self).__init__()
        self.clip_model = clip_model
        self.num_views = num_views
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, images, data_dict):
        if "clip_embeddings_img" in data_dict:
            output = data_dict["clip_embeddings_img"]
        # else:
        #     output = self.clip_model.encode_image(images).reshape(-1, self.num_views, self.clip_model.visual.output_dim)
        #     output = torch.mean(output, dim=1)
        #     output /= output.norm(dim=1, keepdim=True)
        return self.mlp(output)
