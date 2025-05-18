import torch
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self):

        self.w = torch.tensor(1.0, requires_grad=True)
        self.b = torch.tensor(1.0, requires_grad=True)

    def forward(self, x):

        return self.w * x + self.b

    def update(self, learning_rate):

        with torch.no_grad():
            self.w -= learning_rate * self.w.grad
            self.b -= learning_rate * self.b.grad

        # self.reset_grad()

    def reset_grad(self):

        self.w.grad.zero_()
        self.b.grad.zero_()

    def criterion(self, y, yp):

        return torch.mean((yp - y) ** 2)

model = RegressionModel()
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
learning_rate = torch.tensor(0.001)
epochs = 100
losses = []

for epoch in range(epochs):

    y_pred = model.forward(x)

    loss = model.criterion(y, y_pred)
    losses.append(loss.item())

    loss.backward()


    model.update(learning_rate)

    model.reset_grad()

    print("Epoch{}: The parameters are w={}, b={}, and loss={}".format(epoch+1,model.w,model.b,loss.item()))

plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()
