import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import spconv.pytorch as spconv


class SparseCNNEncoder(pl.LightningModule):
    def __init__(self, voxel_size, ef_dim, z_dim, out_dim, **kwargs):
        super().__init__()
        self.voxel_size = voxel_size
        self.sparseModel = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=3, out_channels=ef_dim, kernel_size=3, bias=False),
            nn.BatchNorm1d(ef_dim),
            nn.ReLU(inplace=True),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),

            spconv.SubMConv3d(in_channels=ef_dim, out_channels=ef_dim * 2, kernel_size=3, bias=False),
            nn.BatchNorm1d(ef_dim * 2),
            nn.ReLU(inplace=True),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),

            spconv.SubMConv3d(in_channels=ef_dim * 2, out_channels=ef_dim * 4, kernel_size=3, bias=False),
            nn.BatchNorm1d(ef_dim * 4),
            nn.ReLU(inplace=True),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),

            spconv.SubMConv3d(in_channels=ef_dim * 4, out_channels=ef_dim * 8, kernel_size=3, bias=False),
            nn.BatchNorm1d(ef_dim * 8),
            nn.ReLU(inplace=True),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),

            spconv.SubMConv3d(in_channels=ef_dim * 8, out_channels=z_dim, kernel_size=3, bias=False),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(inplace=True),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            spconv.ToDense()
        )

        self.mlp = nn.Sequential(
            nn.Linear(4096, out_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, batch_size):
        x = spconv.SparseConvTensor(x['feats'], x['locs'], [self.voxel_size] * 3, batch_size)
        x = self.sparseModel(x)
        output = x.reshape(x.shape[0], -1)
        #output = self.out(output)
        return F.normalize(self.mlp(output), dim=1)
