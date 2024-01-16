import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F


class BiGRUEncoder(pl.LightningModule):
    def __init__(self, vocab_size, out_dim, **kwargs):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x, data_dict):
        embeddings = torch.transpose(self.embedding_layer(x), 0, 1)
        h0 = torch.zeros(size=(2, embeddings.shape[1], 128), dtype=torch.float32, device=self.device)
        _, hidden = self.gru(embeddings, h0)
        return F.normalize(torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))), dim=1)
