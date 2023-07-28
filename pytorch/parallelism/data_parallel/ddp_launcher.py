import torch.distributed as dist
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.data.utils import DistributedDataSampler

from torch.data.utils import DataLoader


# process group 셋팅
dist.init_process_group("nccl")
# from torch.distributed import Backend
# dist.init_process_group(backend=Backend.NCCL, init_method='env://')

# 런처에선 각 프로세스에 맞게 알아서 rank를 얻을 수 있고 cuda device를 설정할 수 있음
rank = dist.get_rank()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()

print(rank, device)
# 0 0 2 2 1 1 3 3 이런 식으로 출력됨

world_size = dist.get_world_size()


sampler = DistributedDataSampler(dataset, 
                                 num_replicas=world_size,
                                 rank=rank,
                                 shuffle=True)

data_loader = DataLoader(datasets,
                         batch_size=32,
                         num_workers=4,
                         sampler=sampler,
                         shuffle=False, 
                         pin_memor=True)


model = create_model()
# DDP에선 프로세스마다 각각의 device에 걸리게끔 됨 
model = DistributedDataParallel(model, device_ids = [device], output_device=device)


optimizer = ...

for i, data in enumerate(data_loader):
  
  
                           
                           
                        