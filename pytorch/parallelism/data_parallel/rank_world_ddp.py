import torch.distributed as dist
import torch
import os
import torch.cuda

torch.distributed.init_process_group(backend='nccl', init_method='env://')
# print(os.environ['RANK'])
# print(os.environ['WORLD_SIZE'])
# print(os.environ['LOCAL_RANK'])
# print(torch.cuda.devcie_count())

if __name__ == '__main__':
  print('GLOBAL_RANK', os.environ['RANK']) # str으로 나옴
  print('WORLD_SIZE',os.environ['WORLD_SIZE']) # str으로 나옴
  print('LOCAL_RANK',os.environ['LOCAL_RANK']) # str으로 나옴
  print('DEVICE COUNT', torch.cuda.device_count())
  print(type(os.environ['LOCAL_RANK']))


  
  # dist.get_*()는 init_process_group을 해야 할 수 있음
  print('dist.get_rank', dist.get_rank()) # global_rank를 얻을 수 있음 [0, world_size]
  print('dist.get_world_size', dist.get_world_size())