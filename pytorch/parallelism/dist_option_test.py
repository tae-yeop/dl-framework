import torch.distributed as dist


if __name__ == '__main__':
  dist.init_process_group("nccl")
  print(dist.get_world_size())
  print(dist.get_device_size())