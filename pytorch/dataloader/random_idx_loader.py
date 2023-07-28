import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

import torch.distributed as dist
import os

torch.manual_seed(777)

dist.init_process_group(backend='nccl')

class RandomDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(1.0, 10.0, step=0.2)
        # print(self.data)
        self.data2 = self.data + 0.3
        self.num_variations = 5
        
    def __len__(self):
        return len(self.data)//self.num_variations

    def __getitem__(self, idx):
        # print('idx', idx)
        candidate = [(self.num_variations*idx) + i for i in range(5)]
        print('candidate', candidate)
        # variation_idx = np.random.permutation([0,1,2,3,4])[:2]
        variation_idx = torch.randperm(self.num_variations)[:2]
        # print('localrank', os.environ['LOCAL_RANK'], 'variation_idx', variation_idx[0], variation_idx[1])
        # print('variation_idx', variation_idx[0], variation_idx[1])
        print('candidate[variation_idx[0]])', candidate[variation_idx[0]])
        print('candidate[variation_idx[1]])', candidate[variation_idx[1]])
        data1_sample = self.data[candidate[variation_idx[0]]]
        data2_sample = self.data2[candidate[variation_idx[1]]]

        return data1_sample, data2_sample



class RandomDataset2(Dataset):
    def __init__(self):
        self.data = torch.arange(1.0, 10.0, step=0.2)
        # print(self.data)
        self.data2 = self.data + 0.3
        self.num_variations = 5
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('idx', idx)
        id_idx = idx // self.num_variations
        candidate = id_idx*self.num_variations + torch.randperm(self.num_variations)
        candidate = candidate[candidate!=idx]
        # print(idx, candidate)
        # candidate = [(self.num_variations*idx) + i for i in range(5)]
        # print('candidate', candidate)
        # variation_idx = np.random.permutation([0,1,2,3,4])[:2]
        # variation_idx = torch.randperm(self.num_variations)[:2]
        # print('localrank', os.environ['LOCAL_RANK'], 'variation_idx', variation_idx[0], variation_idx[1])
        # print('variation_idx', variation_idx[0], variation_idx[1])
        # print('candidate[variation_idx[0]])', candidate[variation_idx[0]])
        # print('candidate[variation_idx[1]])', candidate[variation_idx[1]])
        data1_sample = self.data[idx]
        data2_sample = self.data2[candidate[0]]

        return data1_sample, data2_sample


from itertools import combinations
class RandomDataset3(Dataset):
    def __init__(self):
        self.data = torch.arange(1.0, 10.0, step=0.2)
        # print(self.data)
        # self.data2 = self.data + 0.3
        self.num_variations = 5
        
        self.combi = list(combinations([i for i in range(self.num_variations)], 2))
        self.data_indicies = []
        num_total_id_imgs = len(self.data)//self.num_variations
        # print(len(self.data))
        # print(num_total_id_imgs)
        for i in range(num_total_id_imgs):
            subject_idx = i
            for j in self.combi:
                self.data_indicies.append([subject_idx, j[0], j[1]])
                
        
    def __len__(self):
        return len(self.data_indicies)

    def __getitem__(self, idx):
        subject_idx, first_idx, second_idx = self.data_indicies[idx]
        # print('idx', idx, 'subject_idx', subject_idx)
    
        data1_sample = self.data[subject_idx*self.num_variations+first_idx]
        data2_sample = self.data[subject_idx*self.num_variations+second_idx]

        return data1_sample, data2_sample
# print(len(torch.arange(1.0, 100.0, step=0.2)))
# print(torch.randperm(5)[:2])

# print([i for i in range(5)])
# print(len(torch.arange(1.0, 100.0, step=0.2)))

# local_rank = int(os.environ['LOCAL_RANK'])
# mydataset = RandomDataset()
# print(len(mydataset))
# dl = DataLoader(mydataset, batch_size=2, shuffle=False, sampler=DistributedSampler(mydataset))
# dl = DataLoader(mydataset, batch_size=4, shuffle=True)

# print(len(torch.arange(1.0, 10.0, step=0.2))//5)
# print(len(dl))
# for idx, data in enumerate(dl):
#     print(data)
    # print(f'Local Rank : {local_rank}, Idx : {idx}, Data : {data}')


# num_epoch = 2
# for i in range(num_epoch):
#     dl.sampler.set_epoch(i)
#     for j, data in enumerate(dl):
#         print(f'Local Rank : {local_rank}, epcoh: {i}, Idx : {j}, Data : {data}')
#     print('---------------------')


# print(np.random.permutation([0,1,2,3,4])[:2])


# a = torch.randperm(5)
# print(a)
# a = a[a!=3]
# print(a)

# #########################################
# # RandomDataset2
# # getitem에 randperm을 애초에 넣으면 안된다
# # 매번 랜덤하게 대응되는 쌍이 결정되버린다.
# #########################################
# mydataset = RandomDataset2()
# sampler = DistributedSampler(mydataset)
# # dl = DataLoader(mydataset, batch_size=4, shuffle=False, num_workers=4, sampler=sampler)
# dl = DataLoader(mydataset, batch_size=4, shuffle=False, num_workers=4)
# local_rank = int(os.environ['LOCAL_RANK'])
# num_epoch = 2
# for i in range(num_epoch):
#     # dl.sampler.set_epoch(i)
#     for idx, data in enumerate(dl):
#         print(f'Local Rank : {local_rank}, epoch : {i}, idx : {idx}, Data : {data}')
#     print("-"*25)


# #########################################
# # RandomDataset3
# # ctor에서 모든 경우의 수를 미리 만들어서 들고 있는 경우
# # 일단 shuffle하지 말고 nproc=1
# #########################################
# mydataset = RandomDataset3()
# # print(len(mydataset))
# # print(mydataset.data_indicies)
# # sampler = DistributedSampler(mydataset)
# dl = DataLoader(mydataset, batch_size=4, shuffle=False, num_workers=4)#
# local_rank = int(os.environ['LOCAL_RANK'])
# num_epoch = 2
# for i in range(num_epoch):
#     # dl.sampler.set_epoch(i)
#     for idx, data in enumerate(dl):
#         print(f'Local Rank : {local_rank}, epoch : {i}, idx : {idx}, Data : {data}')
#     print("-"*25)


# #########################################
# # RandomDataset3
# # ctor에서 모든 경우의 수를 미리 만들어서 들고 있는 경우
# # 멀티로 해도 의도한대로 돌아간다.
# #########################################
# mydataset = RandomDataset3()
# # print(len(mydataset))
# # print(mydataset.data_indicies)
# sampler = DistributedSampler(mydataset)
# dl = DataLoader(mydataset, batch_size=4, shuffle=False, num_workers=4, sampler=sampler)
# local_rank = int(os.environ['LOCAL_RANK'])
# num_epoch = 2
# for i in range(num_epoch):
#     dl.sampler.set_epoch(i)
#     for idx, data in enumerate(dl):
#         print(f'Local Rank : {local_rank}, epoch : {i}, idx : {idx}, Data : {data}')
#     print("-"*25)



# #########################################
# # RandomDataset3
# # generator를 사용했을 때도 의도한대로 되는지
# # 이 경우는 set_epoch 적용하지 않았을 시
# # local rank 마다 i 번째 epoch에 대해서 j번째 iter는 같은 데이터가 샘플링된다.
# #########################################
# # start
# # start
# # 0 1 [tensor([5.2000, 5.4000, 9.0000, 7.0000]), tensor([5.4000, 5.6000, 9.8000, 7.4000])]
# # [tensor([8.0000, 6.4000, 4.6000, 1.0000]), tensor([8.2000, 6.6000, 4.8000, 1.8000])]
# # start
# # start
# # 1 [tensor([8.0000, 6.4000, 4.6000, 1.0000]), tensor([8.2000, 6.6000, 4.8000, 1.8000])]
# # 0 [tensor([5.2000, 5.4000, 9.0000, 7.0000]), tensor([5.4000, 5.6000, 9.8000, 7.4000])]
# #########################################
# def sample_data(loader):
#   while True:
#     print('start')
#     for batch in loader:
#       yield batch

# mydataset = RandomDataset3()
# # print(len(mydataset))
# # print(mydataset.data_indicies)
# sampler = DistributedSampler(mydataset)
# dl = DataLoader(mydataset, batch_size=4, shuffle=False, num_workers=4, sampler=sampler)
# local_rank = int(os.environ['LOCAL_RANK'])
# # print(len(dl)) # 12
# dl = sample_data(dl)

# i = 0
# bucket = []

# while True:
#     if i == 13:
#         break

#     data = next(dl)
#     if i == 0 or i == 12:
#         print(local_rank, data)
#         bucket.append(data)
#     i += 1


#########################################
# RandomDataset3
# Iterloader 사용
#########################################
# 1 [tensor([8.0000, 6.4000, 4.6000, 1.0000]), tensor([8.2000, 6.6000, 4.8000, 1.8000])]
# 0 [tensor([5.2000, 5.4000, 9.0000, 7.0000]), tensor([5.4000, 5.6000, 9.8000, 7.4000])]
# stop iteration
# stop iteration
# 1 [tensor([8.0000, 7.4000, 7.0000, 8.6000]), tensor([8.8000, 7.8000, 7.8000, 8.8000])]
# 0 [tensor([9.2000, 7.0000, 4.2000, 5.0000]), tensor([9.6000, 7.6000, 4.6000, 5.4000])]
#########################################
# 제대로 나온다.
#########################################
class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        """The number of current epoch.
        Returns:
            int: Epoch number.
        """
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            print('stop iteration')
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

mydataset = RandomDataset3()
sampler = DistributedSampler(mydataset)
dl = DataLoader(mydataset, batch_size=4, shuffle=False, num_workers=4, sampler=sampler)
local_rank = int(os.environ['LOCAL_RANK'])
# print(len(dl)) # 12
dl = IterLoader(dl)

i = 0
bucket = []

while True:
    if i == 13:
        break

    data = next(dl)
    if i == 0 or i == 12:
        print(local_rank, data)
        bucket.append(data)
    i += 1