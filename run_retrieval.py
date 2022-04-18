import os
import sys
import yaml
import argparse

import torch

from tricolo.trainers.SimCLR import SimCLR
from tricolo.dataloader.dataloader import ClrDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, help="Path to config file")
args = parser.parse_args()

def main():
    config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)

    dataset = ClrDataLoader(config['dset'], config['batch_size'], config['sparse_model'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()

if __name__ == "__main__":
    main()
