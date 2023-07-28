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
  

class GatherLayer(torch.autograd.Function):
    """
    https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/losses.py
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
  

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

    if rank == 0:
      sb_model = SubModel().to(device_id)


    # 모든 device로 intermediate result가 들어감
    # device 0: [8,2700] + device 1: [8,2700] => cat => [16,2700]
    gathered_result = torch.cat(GatherLayer.apply(ddp_model.module.intermediate_result), dim=0)
    print(f"rank : {rank} , gathered_result : {gathered_result}")

    if rank == 0:
      sb_out = sb_model(gathered_result)
      print(f'sb_out : {sb_out.shape}')

  

if __name__ == "__main__":
    demo_basic()