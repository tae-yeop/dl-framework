import torch
import torch.nn as nn
import accelerate
from accelerate import Accelerator


class MyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lin = nn.Linear(30, 40)
        self.tensor = torch.randn(1,30).to(device)

    def forward(self, x):
        return self.lin(x)

    def device(self):
        return next(self.lin.parameters()).device

accelerator = Accelerator()
model = MyModel(device=accelerator.device)
model = accelerator.prepare(model)

if accelerator.is_last_process:
    print('tensor device : ', model.module.tensor.device, 'model device : ', model.module.device(), 'accelerator device :', accelerator.device)