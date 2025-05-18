import torch

x = torch.tensor(1.0,requires_grad=True)

y = 8*x**4 + 3*x**3 + 7*x**2 + 6*x + 3

analytical_grad = 32*x**3 + 9*x**2 + 14*x + 6

y.backward()

print("PyTorch grad dy/dx:",x.grad.item())
print("Analytical grad dy/dx:",analytical_grad.item())
