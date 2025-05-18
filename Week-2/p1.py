# x = 2a + 3b
# y = 5a^2 + 3b^3
# z = 2x + 3y

import torch

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0,requires_grad=True)

x = 2*a + 3*b
y = 5*(a**2) + 3*(b**3)
z = 2*x + 3*y

x.retain_grad()
y.retain_grad()

z.backward()

print("a.grad = ", a.grad.item())
print("b.grad = ", b.grad.item())
print("x.grad = ", x.grad.item())
print("y.grad = ", y.grad.item())

analytical_grad = 30*a + 4
print("Analytical gradient dz/da:",analytical_grad.item())




