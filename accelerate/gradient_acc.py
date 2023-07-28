from accelerate import Accelerator
from torchvision.datasets import MNIST

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

transform = Compose([ToTensor(), nn.Flatten()])
dataset = MNIST(root='/home/aiteam/tykim/dl-framework/accelerate', train=True, download=True, transform=transform)
dl = DataLoader(dataset, batch_size=32)


accelerator = Accelerator(gradient_accumulation_steps=5)
  
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(784, 20)
    
  def forward(self, x):
    return self.layer1(x)
  
  

model = Model()
model, dl = accelerator.prepare(model, dl)
for idx, (x,y) in enumerate(dl):

  with accelerator.accumulate(model):
    pred = model(x)
    
    print('ga inner : ', x.shape)
    
  print('ga outer : ',x.shape)
  
