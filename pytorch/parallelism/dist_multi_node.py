# https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide

import os
import argparse

import torch
import torch.distributed as dist


LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    print(int(os.environ['LOCAL_RANK']))
    tensor = torch.zeros(1)

    # Need to put tensor on a GPU device for nccl backend
    if args.backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))

    else:
        dist.recv(tensor=tensor, src=0)
        print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))