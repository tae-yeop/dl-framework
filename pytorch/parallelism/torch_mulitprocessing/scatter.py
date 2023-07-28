import torch
import torch.distributed as dist

dist.init_process_group("gloo")

rank = dist.get_rank()
torch.cuda.set_device(rank)


output = torch.zeros(1)
print(f"before rank {rank}: {output}\n")
# 0번째 프로세스에서 inputs 텐서를 갈라서 scattering함
if rank == 0:
    inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
    inputs = torch.split(inputs, dim=0, split_size_or_sections=1)
		# output 변수에 (tensor([10]), tensor([20]), tensor([30]), tensor([40]))을 
		# 프로세스별로 뿌려서 받게 됨
    dist.scatter(output, scatter_list=list(inputs), src=0)


print(f"after rank {rank}: {output}\n")