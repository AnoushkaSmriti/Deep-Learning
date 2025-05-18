import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class LinearRegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # super(RegressionModel,self).__init__()

        self.linear = nn.Linear(1, 1)  # Single input and output

    def forward(self, x):
        return self.linear(x)

x_data = torch.tensor([[5.0], [7.0], [12.0], [16.0], [20.0]])
y_data = torch.tensor([[40.0], [120.0], [180.0], [210.0], [240.0]])
dataset = LinearRegressionDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = RegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
losses = []

for epoch in range(100):
    epoch_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    losses.append(epoch_loss / len(dataloader))


plt.plot(range(100), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()

with torch.no_grad():  # No need to track gradients for inference

    y_range = model(x_data)


# Plot the original data points

plt.scatter(x_data.numpy(), y_data.numpy(), color='red', label='Data Points')

# Plot the regression line

plt.plot(x_data.numpy(), y_range.numpy(), color='blue', label='Regression Line')

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Linear Regression Fit')

plt.legend()

plt.show()
