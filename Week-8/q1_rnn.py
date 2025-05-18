import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("/home/student/Downloads/daily.csv")

# Preprocess the data - Drop NA values in the dataset
df = df.dropna()
y = df['Price'].values
print(len(y))

# Normalize the input range between 0 and 1
minm = y.min()
maxm = y.max()
print(minm, maxm)
y = (y - minm) / (maxm - minm)

# Sequence length for RNN
Sequence_Length = 10
X = []
Y = []

# Create sequences
for i in range(len(y) - Sequence_Length):
    X.append(y[i:i + Sequence_Length])  # Last 10 days
    Y.append(y[i + Sequence_Length])  # Price for the 11th day

# Convert from list to array
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)


# Define the dataset class
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


# Create dataset and dataloader
dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(dataset, shuffle=True, batch_size=256)


# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _status = self.rnn(x)
        output = output[:, -1, :]  # Get the last time step
        output = self.fc1(torch.relu(output))
        return output


# Initialize the model, optimizer, and loss function
model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 2500
for i in range(epochs):
    for j, data in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        y_pred = model(data[0].view(-1, Sequence_Length, 1)).reshape(-1)
        loss = criterion(y_pred, data[1])
        loss.backward()
        optimizer.step()

    if i % 50 == 0:
        print(f"{i}th iteration: Loss = {loss.item()}")

# Test set actual vs predicted
test_set = NGTimeSeries(x_test, y_test)
test_pred = model(test_set.x.view(-1, Sequence_Length, 1)).view(-1)

# Plot predicted vs original
plt.plot(test_pred.detach().numpy(), label='Predicted')
plt.plot(test_set.y.view(-1).numpy(), label='Original')
plt.legend()
plt.show()

# Undo normalization for final comparison
y = y * (maxm - minm) + minm
y_pred = test_pred.detach().numpy() * (maxm - minm) + minm

# Plot the full series with predictions
plt.plot(y, label='Original Price')
plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label='Predicted Price')
plt.legend()
plt.show()

# Predict the price for the next day based on the last 10 days of the dataset
last_10_days = torch.tensor(y[-Sequence_Length:], dtype=torch.float32).view(1, Sequence_Length, 1)  # Last 10 days
predicted_price = model(last_10_days).item() * (maxm - minm) + minm  # Undo normalization
# predicted_price = model(last_10_days).item()
print(f"Predicted price for the 11th day: {predicted_price:.2f}")