import os
import torch
import torch.distributed as dist

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def pre_forward(self, x):
      return self.relu(self.net1(x))
    def forward(self, x):
        return self.net2(self.pre_forward(x))
    
    def device(self):
        # 그냥 실행하면 일단 모두 cpu에 올라가 있음
        # print('dsad')
        return next(self.net1.parameters()).device

    


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    
    # Model
    
    model = ToyModel()
    # model.requires_grad_(True)
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    
    print(os.environ['LOCAL_RANK'])
    device = "cuda"
    torch.cuda.set_device(local_rank)
    
    
    model = model.to(device)
    model = model.to(local_rank)
    print(model.device())
    
    
    # # DDP Wrapper
    # ddp = DDP(model.to(local_rank), device_ids=[local_rank])#, output_device=local_rank)
    
    # # ddp.requires_grad_(False)
    # model.requires_grad_(False)
    # # print(list(model.parameters()))
    
    # print(list(ddp.parameters()))
    with torch.no_grad():
        print(torch.is_grad_enabled())