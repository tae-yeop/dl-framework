import torch
# x=torch.ones(5,requires_grad=True)
 
# y=x*2
 
# z=x.detach()*3
 
# sum=(y+z).sum()
 
# sum.backward()
 
# print(x.grad) #tensor([2., 2., 2., 2., 2.])
# print(y.grad_fn.next_functions)



a = torch.ones(5, requires_grad=True)
 
b = a**2
c = a.detach()
c.zero_()

print(b.is_leaf)
print(b.grad_fn.next_functions)
print(c.grad_fn)

d = b.sum()
print(d.grad_fn)
print(d.grad_fn.next_functions)

print(a._version)
print(b._version)
print(c._version)
print(d._version)
d.backward()
 
print(a.grad)