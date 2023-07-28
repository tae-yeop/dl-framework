import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 3, kernel_size=3)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    
    self.linear1 = nn.Linear(2700, 10)
    self.relu2 = nn.ReLU()
    self.linear2 = nn.Linear(10, 10)
  def forward(self, x):
    # [B, 2700]
    self.intermediate_result = self.flatten(self.relu(self.conv1(x)))
    result = self.linear2(self.relu2(self.linear1(self.intermediate_result)))
    return result

class SubModel(nn.Module):
  def __init__(self):
    super(SubModel, self).__init__()
    self.linear = nn.Linear(2700, 10)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(10, 3)

  def forward(self, x):
    return self.linear2(self.relu(self.linear(x)))
  

def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    output = ddp_model(torch.randn(8, 3,32, 32))
    print(ddp_model.module.intermediate_result)

    sb_model = SubModel().to(device_id)
    ddp_sb_model = DDP(sb_model, device_ids=[device_id])
    sb_output = ddp_sb_model(ddp_model.module.intermediate_result)
    print(sb_output)
  

if __name__ == "__main__":
    demo_basic()