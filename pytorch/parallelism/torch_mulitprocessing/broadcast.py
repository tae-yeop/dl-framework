import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
# device를 setting하면 이후에 rank에 맞는 디바이스에 접근 가능

# 0번 rank에만 랜덤값
if rank == 0:
    value = torch.randn(2, 2).to(torch.cuda.current_device())
# 나머지는 0 텐서
else:
    value = torch.zeros(2, 2).to(torch.cuda.current_device())

print(f"before rank {rank}: {value}\n")
dist.broadcast(tensor=value, src=0)
print(f"after rank {rank}: {value}\n")