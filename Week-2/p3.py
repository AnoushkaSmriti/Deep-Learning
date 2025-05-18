import torch

b = torch.tensor(1.0,requires_grad=True)
x = torch.tensor(3.0,requires_grad=True)
w = torch.tensor(2.0,requires_grad=True)

u = w*x
v = u+b
a = torch.sigmoid(v)

a.backward()

analytical_grad = torch.sigmoid(v)*(1-torch.sigmoid(v))*x

print("w.grad: ",w.grad.item())
print("Analytical grad da/dw: ",analytical_grad.item())
