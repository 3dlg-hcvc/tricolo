"""
Code modified from: https://github.com/edreisMD/ConVIRT-pytorch/blob/master/models/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tricolo.models.models import cnn_encoder, cnn_encoder32, cnn_encoder_sparse, SVCNN, MVCNN

class ModelCLR(nn.Module):
    def __init__(self, dset, voxel_size, sparse_model, out_dim, use_voxel, tri_modal, num_images, image_cnn, pretraining, vocab_size):
        super(ModelCLR, self).__init__()

        self.dset = dset
        self.ef_dim = 32
        self.z_dim = 512
        self.out_dim = out_dim
        self.cnn_name = image_cnn
        self.use_voxel = use_voxel
        self.tri_modal = tri_modal
        self.voxel_size = voxel_size
        self.num_images = num_images
        self.pretraining = pretraining
        self.sparse_model = sparse_model
        
        self.text_model, self.text_fc = self._get_text_encoder()
        self.embedding_layer = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.voxel_model, self.voxel_fc, self.image_model, self.image_fc = self._get_res_encoder()

    def _get_text_encoder(self):
        print("Text feature extractor: BiGRU")
        text_model = nn.GRU(input_size=256, hidden_size=128, num_layers=1, bidirectional=True)
        text_fc = nn.Linear(256, self.out_dim)
        return text_model, text_fc

    def _get_res_encoder(self):
        voxel_model = None
        voxel_fc = None
        image_model = None
        image_fc = None

        if self.dset == 'shapenet':
            if self.tri_modal:
                print('Training Tri-Modal Model')
                if self.sparse_model:
                    voxel_model = cnn_encoder_sparse(self.voxel_size, self.ef_dim, self.z_dim)
                else:
                    voxel_model = cnn_encoder(self.voxel_size, self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))

                svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
            elif self.use_voxel:
                print('Training Bi-Modal Model')
                if self.sparse_model:
                    voxel_model = cnn_encoder_sparse(self.voxel_size, self.ef_dim, self.z_dim)
                else:
                    voxel_model = cnn_encoder(self.voxel_size, self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
            else:
                print('Training Bi-Modal Model')
                svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
        elif self.dset == 'primitives':
            print('Training Primitives')
            if self.tri_modal:
                raise('Implement Other Dataset')
            elif self.use_voxel:
                voxel_model = cnn_encoder32(self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                print('Bi-Modal Voxel, Text')
            else:
                raise('Implement Other Dataset')
        else:
            raise('Implement Other Dataset')
        return voxel_model, voxel_fc, image_model, image_fc

    def voxel_encoder(self, xis):
        h = self.voxel_model(xis)
        h.squeeze()
        x = self.voxel_fc(h)
        return x

    def image_encoder(self, xis):
        h = self.image_model(xis)
        h.squeeze()
        x = self.image_fc(h)
        return x

    def text_encoder(self, encoded_inputs):
        embed_inputs = self.embedding_layer(encoded_inputs)
        embed_inputs = torch.transpose(embed_inputs, 0, 1)

        N = embed_inputs.shape[1]

        h0 = torch.zeros(2, N, 128).cuda()
        output, hidden = self.text_model(embed_inputs, h0)
        out_emb = torch.tanh(self.text_fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return out_emb

    def forward(self, voxels, images, encoded_inputs):
        z_voxels = None
        z_images = None
        if self.tri_modal:
            images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
            z_voxels = self.voxel_encoder(voxels)
            z_images = self.image_encoder(images)
        elif self.use_voxel:
            z_voxels = self.voxel_encoder(voxels)
        else:
            images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
            z_images = self.image_encoder(images)

        zls = self.text_encoder(encoded_inputs)
        return z_voxels, z_images, zls
