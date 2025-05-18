import torch
import matplotlib.pyplot as plt

x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9,
                  18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2,
                  17.4, 19.5, 19.7, 21.2], dtype=torch.float32)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9,
                  15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1,
                  16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
# lr = torch.tensor(0.001)
lr = 0.001
losses = []
epochs = 10

for epoch in range(epochs):
    y_pred = w*x+b
    loss = torch.mean((y_pred-y)**2)

    loss.backward()
    # losses.append(loss.item())
    print(f"Epoch{epoch+1}:")
    print(f"w.grad:{w.grad.item()}")
    print(f"b.grad:{b.grad.item()}")



    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad

        w.grad.zero_()
        b.grad.zero_()

    losses.append(loss.item())
# print(losses)
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()
