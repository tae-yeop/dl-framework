import torch

x=torch.ones(5,requires_grad=True)
 
y=x.clone()*2
 
z=x.clone()*3
 
sum=(y+z).sum()
 
sum.backward()

print(y.is_leaf)

print(x.grad)
# intermeditate tensor의 grad를 보존되지 않음
print(y.grad)