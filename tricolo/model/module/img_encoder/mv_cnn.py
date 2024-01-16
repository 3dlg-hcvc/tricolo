import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class MVCNNEncoder(pl.LightningModule):
    """
    From https://github.com/jongchyisu/mvcnn_pytorch/blob/master/models/MVCNN.py
    """
    def __init__(self, z_dim, out_dim, cnn_name, num_views, **kwargs):
        super(MVCNNEncoder, self).__init__()
        svcnn_model = SVCNN(z_dim, cnn_name=cnn_name)
        self.num_views = num_views
        if svcnn_model.efficientnet:
            self.net_1 = svcnn_model
        else:
            self.net_1 = nn.Sequential(*list(svcnn_model.net.children())[:-1])
        self.net_2 = svcnn_model.net.fc
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, data_dict):
        y = self.net_1(x)
        y = y.view((x.shape[0] // self.num_views, self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
        y = self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
        y.squeeze()
        return F.normalize(self.mlp(y), dim=1)


class SVCNN(pl.LightningModule):
    """
    From https://github.com/jongchyisu/mvcnn_pytorch/blob/master/models/MVCNN.py
    """
    def __init__(self, z_dim, cnn_name):
        super().__init__()
        self.efficientnet = False
        if cnn_name == 'resnet18':
            self.net = models.resnet18(weights="ResNet18_Weights.DEFAULT")
            self.net.fc = nn.Linear(512, z_dim)
        elif cnn_name == 'resnet34':
            self.net = models.resnet34(weights="ResNet34_Weights.DEFAULT")
            self.net.fc = nn.Linear(512, z_dim)
        elif cnn_name == 'resnet50':
            self.net = models.resnet50(weights="ResNet50_Weights.DEFAULT")
            self.net.fc = nn.Linear(2048, z_dim)
        elif cnn_name == 'efficientnet_b0':
            self.net = EfficientNet.from_pretrained('efficientnet-b0')
            self.net.fc = nn.Linear(1280, z_dim)
            self.efficientnet = True
        elif cnn_name == 'efficientnet_b3':
            self.net = EfficientNet.from_pretrained('efficientnet-b3')
            self.net.fc = nn.Linear(1536, z_dim)
            self.efficientnet = True

    def forward(self, x):
        if self.efficientnet:
            output = self.net.extract_features(x)
            output = self.net._avg_pooling(output)
            return output
        else:
            return self.net(x)
