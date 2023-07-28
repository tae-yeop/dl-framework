from argparse import ArgumentParser
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed import Backend


import random
import numpy as np
import os
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    
def create_data_loaders(rank, world_size, batch_size) -> Tuple[DataLoader, DataLoader]:
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
  dataset_loc = './mnist_data'

  train_dataset = datasets.MNIST(dataset_loc, download=True, train=True,
                                 transform=transform)

  sampler = DistributedSampler(train_dataset, 
                               num_replicas=world_size, # <-- world size만큼
                               rank=rank, # <-- 보통 0번째 device의 rank가 들어감
                               shuffle=True, # <-Must be True
                               seed=42)
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size, 
                            shuffle=False, # <- Must be False
                            num_workers=4,
                            sampler=sampler,
                            pin_memory=True)

  # test와 val은 distrbuted가 필요하지 않다.
  test_dataset = datasets.MNIST(dataset_loc,
                                download=True,
                                train=False,
                                transform=transform)
  test_loader = DataLoader(test_dataset,
                           batch_size = batch_size,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True)
  return train_loader, test_loader



def create_model():
  # create model architecture
  model = nn.Sequential(
      nn.Linear(28*28, 128),  # MNIST images are 28x28 pixels
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 10, bias=False)  # 10 classes to predict
  )
  return model

def main(rank, epochs, model, train_loader, test_loader) -> nn.Module:
  
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  loss = nn.CrossEntropyLoss()

  for i in range(epochs):
    model.train()
    train_loader.sampler.set_epoch(i)

    epoch_loss = 0
    pbar = tqdm(train_loader)
    for x, y in pbar:
      print('배치 사이즈 : ', x.shape[0])
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)
      x = x.view(x.shape[0], -1)
      
      optimizer.zero_grad()
      y_hat = model(x)

      batch_loss = loss(y_hat, y)
      batch_loss.backward()

      optimizer.step()

      batch_loss_scalar = batch_loss.item()
      epoch_loss += batch_loss_scalar / x.shape[0]
      pbar.set_description(f'training batch loss={batch_loss_scalar:.4f}')

    # epoch 끝날 때 마다 validatiaon
    with torch.no_grad():
      model.eval()
      val_loss = 0
      pbar = tqdm(test_loader)
      for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.view(x.shape[0], -1)
        y_hat = model(x)
        batch_loss = loss(y_hat, y)
        batch_loss_scalar = batch_loss.item()
        val_loss += batch_loss_scalar / x.shape[0]
        pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')

    print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

  return model.module



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--local_rank', type=int)
  parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
  parser.add_argument("--batch_size", type=int, help="Training batch size for one process.")
  parser.add_argument("--learning_rate", type=float, help="Learning rate.")
  parser.add_argument("--random_seed", type=int, help="Random seed.", default=0)
  parser.add_argument("--model_dir", type=str, help="Directory for saving models.")
  parser.add_argument("--model_filename", type=str, help="Model filename.")
  parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
  
  args = parser.parse_args()

  local_rank = args.local_rank
  num_epochs = args.num_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  random_seed = args.random_seed
  model_dir = args.model_dir
  model_filename = args.model_filename
  resume = args.resume

  
  batch_size = 128
  epochs = 1
  set_random_seeds(random_seed=random_seed)
  
  rank = args.local_rank
  world_size = torch.cuda.device_count()
  
  torch.distributed.init_process_group(backend=Backend.NCCL,
                                       init_method='env://')
  torch.cuda.set_device(rank)

  train_loader, test_loader = create_data_loaders(rank, world_size, batch_size)

  if model_dir is not None and model_filename is not None:
    model_filepath = os.path.join(model_dir, model_filename)

  device = torch.device(f'cuda:{rank}')
  model = create_model()
  model = model.to(device)
  model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

  if resume == True:
    map_location = {"cuda:0": "cuda:{}".format(rank)}
    model.load_state_dict(torch.load(model_filepath, map_location=map_location))
        
  model = main(rank=rank, epochs=epochs, model=model, 
               train_loader=train_loader, test_loader=test_loader)


  if rank == 0:
    torch.save(model.state_dict(), 'model.pt')