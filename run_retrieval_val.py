import os
import sys
import yaml
import json
import random
import argparse
import numpy as np

import torch

from tricolo.trainers.SimCLR import SimCLR
from tricolo.dataloader.dataloader import ClrDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="None", help="Exp to evaluate")
parser.add_argument("--split", type=str, help="Dataset split to evaluate on (valid or test)")
parser.add_argument('--clip', action='store_true', help='Use pretrained CLIP to evaluate')
args = parser.parse_args()

def main(load_dir):
    if not args.clip:
        with open(load_dir + '/checkpoints/config.json', 'r') as f:
            config = json.load(f)
        config['train'] = False
        config['log_dir'] = load_dir
    else:
        "Dummy config file"
        config = yaml.load(open('./tricolo/configs/clip.yaml', "r"), Loader=yaml.FullLoader)
        config['train'] = False
        config['log_dir'] = './logs/retrieval/clip'

    dataset = ClrDataLoader(config['dset'], config['batch_size'], config['sparse_model'], **config['dataset'])
    simclr = SimCLR(dataset, config)

    pr_at_k = simclr.test(config['log_dir'], clip=args.clip, eval_loader=args.split)

    precision = pr_at_k['precision']
    recall = pr_at_k['recall']
    recall_rate = pr_at_k['recall_rate']
    ndcg = pr_at_k['ndcg']
    # r_rank = pr_at_k['r_rank']

    rr_1 = recall_rate[0]
    rr_5 = recall_rate[4]
    ndcg_5 = ndcg[4]

    return rr_1, rr_5, ndcg_5

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    path = './logs/retrieval/' + args.exp
    load_dirs = [path]

    rr_1 = []
    rr_5 = []
    ndcg_5 = []
    print(load_dirs)
    for load_dir in load_dirs:
        _rr_1, _rr_5, _ndcg_5 = main(load_dir)
        torch.cuda.empty_cache()

        rr_1.append(_rr_1)
        rr_5.append(_rr_5)
        ndcg_5.append(_ndcg_5)

    # Report back numbers as percentages
    rr_1 = np.array(rr_1) * 100
    rr_5 = np.array(rr_5) * 100
    ndcg_5 = np.array(ndcg_5) * 100
    print(np.mean(rr_1), np.mean(rr_5), np.mean(ndcg_5))
