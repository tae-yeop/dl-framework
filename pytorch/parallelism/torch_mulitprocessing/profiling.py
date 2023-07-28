import torch
import torch.distributed as dist
with torch.profiler():
    tensor = torch.randn(20, 10)
    dist.all_reduce(tensor)