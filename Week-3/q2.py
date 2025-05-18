import torch
import matplotlib.pyplot as plt

x = torch.tensor([2.0,4.0])
y = torch.tensor([20.0,40.0])

w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0],requires_grad = True)
lr = torch.tensor(0.001)
# lr = 0.001

epochs=2
losses=[]
for i in range(epochs):
    y_pred = w*x + b
    loss = torch.mean((y-y_pred)**2)
    # loss = torch.mean((y_pred-y)**2) #same

    loss.backward()

    print(f"Epoch {i + 1}:")
    print(f"w.grad: {w.grad.item()}")
    print(f"b.grad: {b.grad.item()}")
    print(f"Loss: {loss}")

    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
        print(f"Updated w: {w.item()}, Updated b: {b.item()}\n")


    w.grad.zero_()
    b.grad.zero_()
    # print(f"Updated w: {w.item()}, Updated b: {b.item()}\n")


    losses.append(loss.item())

# print(f"Updated w: {w.item()}, Updated b: {b.item()}\n")

plt.plot(range(epochs),losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.show()

