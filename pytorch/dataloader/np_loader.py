import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os
from torch.utils.data import DistributedSampler

dist.init_process_group(backend='nccl')


class RandomDataset(Dataset):
    def __getitem__(self, index):
        return np.random.randint(0, 1000, 3)

    def __len__(self):
        return 16
    
dataset = RandomDataset()
# dataloader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=True)


# for batch in dataloader:
#     print(batch)

# local_rank = int(os.environ['LOCAL_RANK'])
# for epoch in range(3):
#     for idx,batch in enumerate(dataloader):
#         print(f'Local Rank : {local_rank}, epoch : {epoch}, idx : {idx}, Data : {batch}')
#     print("-"*25)


# 
dataloader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=False, sampler=DistributedSampler(dataset))
local_rank = int(os.environ['LOCAL_RANK'])
for epoch in range(3):
    for idx,batch in enumerate(dataloader):
        print(f'Local Rank : {local_rank}, epoch : {epoch}, idx : {idx}, Data : {batch}')
    print("-"*25)