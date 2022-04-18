import torch
import torch.nn as nn
import sparseconvnet as scn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class cnn_encoder_sparse(nn.Module):
    def __init__(self, voxel_size, ef_dim, z_dim):
        super(cnn_encoder_sparse, self).__init__()

        print('Sparse Voxel Encoder')
        self.ef_dim = ef_dim
        self.z_dim = z_dim

        self.input_layer = scn.InputLayer(3, 64, mode=4)
        self.sparseModel = scn.SparseVggNet(3, 3, [
            ['C', self.ef_dim], ['MP', 2, 2],
            ['C', self.ef_dim*2], ['MP', 2, 2],
            ['C', self.ef_dim*4], ['MP', 2, 2],
            ['C', self.ef_dim*8], ['MP', 2, 2],
            ['C', self.z_dim], ['MP', 2, 2]]
        ).add(scn.SparseToDense(3, self.z_dim))

        self.out = nn.Linear(4096, self.z_dim)
        
    def forward(self,_x):
        x = [_x['locs'], _x['feats']]
        x = self.input_layer(x)
        x = self.sparseModel(x)

        flatten = x.reshape(x.shape[0], -1)
        output = self.out(flatten)
        return output

class cnn_encoder(nn.Module):
    def __init__(self, voxel_size, ef_dim, z_dim):
        super(cnn_encoder, self).__init__()

        print('Dense Voxel Encoder')
        self.ef_dim = ef_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(4, self.ef_dim, 3, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)

        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 3, stride=1, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
        self.pool_2 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 3, stride=1, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
        self.pool_3 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 3, stride=1, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim*8)
        self.pool_4 = nn.MaxPool3d(3, stride=2, padding=1)

        if voxel_size == 32:
            last_stride = 1
        else:
            last_stride = 2

        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 3, stride=last_stride, padding=1, bias=True)
        self.in_5 = nn.InstanceNorm3d(self.z_dim)
        self.pool_5 = nn.AdaptiveAvgPool3d((2,2,2))

        self.out = nn.Linear(4096, self.z_dim)

        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias,0)

    def forward(self, inputs):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)
        d_2 = self.pool_2(d_2)
        
        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)
        d_3 = self.pool_3(d_3)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)
        d_4 = self.pool_4(d_4)

        d_5 = self.in_5(self.conv_5(d_4))
        d_5 = F.leaky_relu(d_5, negative_slope=0.02, inplace=True)
        d_5 = self.pool_5(d_5)

        flatten = d_5.reshape(d_5.shape[0], -1)
        output = self.out(flatten)
        return output

class cnn_encoder32(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(cnn_encoder32, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim

        self.conv_1 = nn.Conv3d(4, self.ef_dim, 3, stride=1, padding=1, bias=False) # 16
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)

        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 3, stride=1, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
        self.pool_2 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 3, stride=1, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
        self.pool_3 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 3, stride=1, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim*8)
        self.pool_4 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 3, stride=2, padding=1, bias=True)
        self.in_5 = nn.InstanceNorm3d(self.z_dim)

        self.out = nn.Linear(4096, self.z_dim)

        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias,0)

    def forward(self, inputs, is_training=False): # inputs: 32, 1, 32, 32, 32
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True) # 32, 32, 32, 32, 32

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True) # 32, 64, 16, 16, 16
        d_2 = self.pool_2(d_2)
        
        d_3 = self.in_3(self.conv_3(d_2)) # 32, 128, 8, 8, 8
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)
        d_3 = self.pool_3(d_3)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True) # 32, 256, 4, 4, 4
        d_4 = self.pool_4(d_4)

        d_5 = self.in_5(self.conv_5(d_4)) # 32, 256, 1, 1, 1
        d_5 = F.leaky_relu(d_5, negative_slope=0.02, inplace=True)

        flatten = d_5.reshape(d_5.shape[0], -1) # 32, 256
        output = self.out(flatten)

        return output

class SVCNN(nn.Module):
    """
    From https://github.com/jongchyisu/mvcnn_pytorch/blob/master/models/MVCNN.py
    """
    def __init__(self, z_dim, pretraining=False, cnn_name='resnet50'):
        super(SVCNN, self).__init__()

        self.z_dim = z_dim
        self.cnn_name = cnn_name
        self.pretraining = pretraining
        self.efficientnet = False

        if self.cnn_name == 'resnet18':
            self.net = models.resnet18(pretrained=self.pretraining)
            self.net.fc = nn.Linear(512, self.z_dim)
        elif self.cnn_name == 'resnet34':
            self.net = models.resnet34(pretrained=self.pretraining)
            self.net.fc = nn.Linear(512, self.z_dim)
        elif self.cnn_name == 'resnet50':
            self.net = models.resnet50(pretrained=self.pretraining)
            self.net.fc = nn.Linear(2048, self.z_dim)
        elif self.cnn_name == 'efficientnet_b0':
            self.net = EfficientNet.from_pretrained('efficientnet-b0')
            self.net.fc = nn.Linear(1280, self.z_dim)
            self.efficientnet = True
        elif self.cnn_name == 'efficientnet_b3':
            self.net = EfficientNet.from_pretrained('efficientnet-b3')
            self.net.fc = nn.Linear(1536, self.z_dim)
            self.efficientnet = True

    def forward(self, x):
        if self.efficientnet:
            output = self.net.extract_features(x)
            output = self.net._avg_pooling(output)
            return output
        else:
            return self.net(x)


class MVCNN(nn.Module):
    """
    From https://github.com/jongchyisu/mvcnn_pytorch/blob/master/models/MVCNN.py
    """
    def __init__(self, z_dim, model, cnn_name='resnet50', num_views=1):
        super(MVCNN, self).__init__()

        print('Image Encoder: {}, Num Views: {}'.format(cnn_name, num_views))
        self.num_views = num_views

        if model.efficientnet:
            self.net_1 = model
        else:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
        self.net_2 = model.net.fc

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return self.net_2(torch.max(y,1)[0].view(y.shape[0],-1))
