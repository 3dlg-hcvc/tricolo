import os
import nrrd
import jsonlines
import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset

class ClrDataset(Dataset):
    def __init__(self, json_file, sparse_model, image_size, voxel_size, root_npz_file='./datasets/all_npz/'):
        self.clr_frame = []
        with jsonlines.open(json_file) as reader:
            for obj in reader:
                self.clr_frame.append(obj)
        self.root_npz_file = root_npz_file
        
        print('Image Resolution: {}, Voxel Resolution: {}'.format(image_size, voxel_size))
        self.image_size = image_size
        self.voxel_size = voxel_size

        self.sparse_model = sparse_model

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        model_id = self.clr_frame[idx]['model']
        category = self.clr_frame[idx]['category']

        path = self.root_npz_file + category + '/' + model_id + '.npz'
        data = np.load(path)

        if self.voxel_size == 32:
            voxel_data = data['voxel32']
        elif self.voxel_size == 64:
            voxel_data = data['voxel64']
        elif self.voxel_size == 128:
            voxel_data = data['voxel128']
        else:
            raise('Not supported voxel size')

        coords, colors = voxel_data
        coords = coords.astype(int)
        voxels = np.zeros((4, self.voxel_size, self.voxel_size, self.voxel_size))
        for i in range(coords.shape[0]):
            voxels[:3, coords[i, 0], coords[i, 1], coords[i, 2]] = colors[i]
            voxels[-1, coords[i, 0], coords[i, 1], coords[i, 2]] = 1

        images = data['images']
        if self.image_size != 224:
            resized = []
            for i in range(images.shape[0]):
                image = images[i].transpose(1, 2, 0)
                image = cv.resize(image, dsize=(self.image_size, self.image_size))
                resized.append(image)
            resized = np.array(resized)
            images = resized.transpose(0, 3, 1, 2)
        
        text = self.clr_frame[idx]['caption']
        text = text.replace("\n", "")

        tokens = np.asarray(self.clr_frame[idx]['arrays'])

        if self.sparse_model:
            grid = np.transpose(voxels, (1, 2, 3, 0))
            grid = grid / 255.
            a, b = [], []
            a = np.array(grid[:, :, :, -1].nonzero()).transpose((1, 0))
            b = grid[a[:, 0], a[:, 1], a[:, 2], :3]
            a = torch.from_numpy(np.array(a)).long()
            b = torch.from_numpy(np.array(b)).float()
            locs = a
            feats = b

            data_dict = {'model_id': model_id,
                        'category': category,
                        'text': text,
                        'tokens': tokens,
                        'images': images.astype(np.float32),
                        'voxels': {'locs': locs, 'feats': feats}}
            return data_dict
        else:
            data_dict = {'model_id': model_id,
                        'category': category,
                        'text': text,
                        'tokens': tokens,
                        'images': images.astype(np.float32),
                        'voxels': voxels.astype(np.float32)}
            return data_dict

class ClrDatasetPrimitives(Dataset):
    def __init__(self, json_file, voxel_root_dir='./datasets/text2shape-data/primitives/primitives.v2'):
        self.clr_frame = []
        with jsonlines.open(json_file) as reader:
            for obj in reader:
                self.clr_frame.append(obj)
        
        self.voxel_root_dir = voxel_root_dir

        print('Using primitives only voxel size 32 and no images supported')

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        model_id = self.clr_frame[idx]['model']
        category = self.clr_frame[idx]['category']

        images = np.zeros((1))
        voxel_name = os.path.join(self.voxel_root_dir, category, model_id+'.nrrd')
        voxel_model, _ = nrrd.read(voxel_name)
        voxels = voxel_model.astype(np.float32)

        voxels = voxels / 255.
        
        text = self.clr_frame[idx]['caption']
        text = text.replace("\n", "")

        category = self.clr_frame[idx]['category']
        tokens = np.asarray(self.clr_frame[idx]['arrays'])

        data_dict = {'model_id': model_id,
                     'category': category,
                     'text': text,
                     'tokens': tokens,
                     'images': images.astype(np.float32),
                     'voxels': voxels.astype(np.float32)}
        return data_dict
        