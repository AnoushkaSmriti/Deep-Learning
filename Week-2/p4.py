import torch

def function(x):
    return torch.exp(-x**2-2*x-torch.sin(x))

def analytical_grad(x):
    f = function(x)
    grad = f * (-2*x-2-torch.cos(x))
    return grad

x_value = 1.0
x = torch.tensor(x_value,requires_grad=True)

f = function(x)
f.backward()

pytorch_grad = x.grad.item()
analytical_grad = analytical_grad(x).item()

print("PyTorch grad df/dx: ",pytorch_grad)
print("Analytical grad df/dx: ",analytical_grad)
print("Difference :",{abs(pytorch_grad-analytical_grad)})

