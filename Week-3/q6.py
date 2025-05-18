import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class multilinearRegression(nn.Module):
    def __init__(self):
        super(multilinearRegression,self).__init__()
        self.linear = nn.Linear(2,1)

    def forward(self, x):
        return self.linear(x)


data = {
    'X1': [3, 4, 5, 6, 2],
    'X2': [8, 5, 7, 3, 1],
    'Y': [-3.7, 3.5, 2.5, 11.5, 5.7]
}

X = torch.tensor(list(zip(data['X1'], data['X2'])), dtype=torch.float32)
# X = torch.tensor([[data['X1']], [data['X2']]], dtype=torch.float32).view(-1,1)   # wrong
y = torch.tensor(data['Y'], dtype=torch.float32).view(-1, 1)

dataset = RegressionDataset(X, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# model = nn.Linear(2, 1)  # 2 input features (X1, X2), 1 output feature (Y)
model = multilinearRegression()
learning_rate = 0.001
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []

for epoch in range(10):
    epoch_loss = 0.0

    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()  # Reset gradients

        y_pred = model(x_batch)  # Forward pass
        loss = criterion(y_pred, y_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    loss_list.append(epoch_loss)

    # print(f"Epoch {epoch + 1}: w={model.weight.data.numpy()}, b={model.bias.data.numpy()}, loss={epoch_loss}")
    print(f"Epoch {epoch + 1}: w={model.linear.weight.data.numpy()}, b={model.linear.bias.data.numpy()}, loss={epoch_loss}")

plt.plot(loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.show()

with torch.no_grad():
    test_input = torch.tensor([[3.0, 2.0]], dtype=torch.float32)
    prediction = model(test_input)
    print(f"Prediction for X1=3, X2=2: {prediction.item()}")
