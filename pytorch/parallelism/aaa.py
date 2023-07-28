from torchvision.models import resnet18
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image

import os
import builtins
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist



model = resnet18(weights=False)

class Dataset(Dataset):
    def __init__(self, root, img_range=(0, 65000), 
                 sub_dir='ffhq',
                 crop=False,
                 resize_size=None,
                 interpolation='lanczos',
                 random_flip=False,
                 normalize=True):
        
        self.root = Path(root, sub_dir)
        
        self.img_p_list = sorted(list(Path('/home/aiteam/tykim/generative/gan/3D-FM-GAN/data/ffhq/images').rglob('*[!_r].png')))[img_range[0]:img_range[1]]


        self.trsf = transforms.Compose([transforms.Resize(resize_size), transforms.ToTensor(), 
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), transforms.ToTensor()])
        
    def __len__(self):
        return len(self.img_p_list)

    def __getitem__(self, idx):
        return self.trsf(Image.open(self.img_p_list[idx]))
      
      

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args
  
def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    model = MyModel()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    ### optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    ### resume training if necessary ###
    if args.resume:
        pass
    
    ### data ###
    train_dataset = MyDataset(mode='train')
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_dataset = MyDataset(mode='val')
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    torch.backends.cudnn.benchmark = True
    
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #
        
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        if args.rank == 0: # only val and save on master node
            validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    pass
    # only one gpu is visible here, so you can send cpu data to gpu by 
    # input_data = input_data.cuda() as normal
    
def validate(val_loader, model, criterion, epoch, args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)