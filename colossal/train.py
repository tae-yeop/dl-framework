import os
import argparse

import colossalai
from colossalai.logging import get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.models as tvm
import torchvision.transforms as transforms
from tqdm import tqdm

trsf = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

USE_MP = True

if __name__ == '__main__':
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    print(args)
    
    # https://colossalai.org/docs/basics/launch_colossalai
    colossalai.launch_from_torch(config='config/colossal_config.py')
    logger = get_dist_logger()

    '''Dataset Init'''
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trsf)
    trainloader = get_dataloader(dataset=trainset, batch_Size, )