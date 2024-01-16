import torch.nn as nn
import lightning.pytorch as pl


class CLIPTextEncoder(pl.LightningModule):
    def __init__(self, out_dim, clip_model):
        super(CLIPTextEncoder, self).__init__()
        self.clip_model = clip_model
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, tokens, data_dict):
        if "clip_embeddings_text" in data_dict:
            output = data_dict["clip_embeddings_text"]
        # else:
        #     output = self.clip_model.encode_text(tokens)
        #     output /= output.norm(dim=1, keepdim=True)
        return self.mlp(output)
