import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()
import torch.profiler

model = models.resnet50(pretrained=True)
model.cuda()
cudnn.benchmark = True
  
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0")
model.train()

def train():
  for step, data in enumerate(trainloader, 0):
      print("step:{}".format(step))
      inputs, labels = data[0].to(device=device), data[1].to(device=device)

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if step + 1 >= 4:
          break
if __name__ == '__main__':
  with torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=True):
    train()