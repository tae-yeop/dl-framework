import copy
import torch
# from torchvision.datasets.mnist import MNIST
# from torc
# MNIST(root='', train=True, transform=)

print(torch.rand(20))
x=torch.ones(5,requires_grad=True)
 
x_deepcopy = copy.deepcopy(x)
 
print(x,'x')
print(x_deepcopy,'x deepcopy')
print(x_deepcopy.grad_fn)


y = x * 2
y_deepcopy = copy.deepcopy(y)

print(y,'y')
print(y_deepcopy,'y deepcopy')
print(y.grad_fn)
print(y_deepcopy.grad_fn)
