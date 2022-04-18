import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from tricolo.dataloader.dataset import ClrDataset, ClrDatasetPrimitives

def collate_fn(batch):
    default_collate_items = ['model_id', 'category', 'text', 'tokens', 'images']

    locs = []
    feats = []
    data = []
    for i, item in enumerate(batch):
        _locs = batch[i]['voxels']['locs']
        locs.append(torch.cat([_locs, torch.LongTensor(_locs.shape[0],1).fill_(i)],1))
        feats.append(batch[i]['voxels']['feats'])

        data.append({k:item[k] for k in default_collate_items})

    locs = torch.cat(locs)
    feats = torch.cat(feats)
    data = default_collate(data)
    data['voxels'] = {'locs': locs, 'feats': feats}
    return data

class ClrDataLoader(object):
    def __init__(self, dset, batch_size, sparse_model, num_workers, train_json_file, val_json_file, test_json_file, image_size, voxel_size, root_npz_file='./datasets/all_npz/'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_json_file = train_json_file
        self.val_json_file = val_json_file
        self.test_json_file = test_json_file
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.sparse_model = sparse_model
        self.root_npz_file = root_npz_file
        self.dset = dset
        
    def get_data_loaders(self):
        if self.dset == 'shapenet':
            print('Using Shapenet Dataset')
            train_dataset = ClrDataset(json_file=self.train_json_file, sparse_model=self.sparse_model, image_size=self.image_size, voxel_size=self.voxel_size, root_npz_file=self.root_npz_file)
            valid_dataset = ClrDataset(json_file=self.val_json_file, sparse_model=self.sparse_model, image_size=self.image_size, voxel_size=self.voxel_size, root_npz_file=self.root_npz_file)
            test_dataset = ClrDataset(json_file=self.test_json_file, sparse_model=self.sparse_model, image_size=self.image_size, voxel_size=self.voxel_size, root_npz_file=self.root_npz_file)
        elif self.dset == 'primitives':
            print('Using Primitives Dataset')
            train_dataset = ClrDatasetPrimitives(json_file=self.train_json_file, voxel_root_dir=self.root_npz_file)
            valid_dataset = ClrDatasetPrimitives(json_file=self.val_json_file, voxel_root_dir=self.root_npz_file)
            test_dataset = ClrDatasetPrimitives(json_file=self.test_json_file, voxel_root_dir=self.root_npz_file)
        else:
            raise('Implement Other Dataset')

        if self.sparse_model:
            train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
            valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
            test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=True)

        print('Training file: {}, Size: {}'.format(self.train_json_file, len(train_loader.dataset)))
        print('Val file: {}, Size: {}'.format(self.val_json_file, len(valid_loader.dataset)))
        print('Test file: {}, Size: {}'.format(self.test_json_file, len(test_loader.dataset)))

        return train_loader, valid_loader, test_loader
