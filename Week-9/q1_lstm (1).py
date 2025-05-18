import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Load the dataset
df = pd.read_csv("/home/student/Downloads/daily.csv")

# Preprocess the data - Drop NA values in the dataset
df = df.dropna()
y = df['Price'].values
x = np.arange(1, len(y), 1)

# Normalize the input range between 0 and 1
minm = y.min()
maxm = y.max()
y = (y - minm) / (maxm - minm)

# Define sequence length
Sequence_Length = 10
X = []
Y = []

# Create sequences
for i in range(len(y) - Sequence_Length):
    list1 = []
    for j in range(i, i + Sequence_Length):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[i + Sequence_Length])  # Predict the next value after the sequence

X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)


# Create a custom dataset
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


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]  # Get the last time step
        output = self.fc1(torch.relu(output))
        return output


# Initialize the model, loss function, and optimizer
model = LSTMModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 500

# Training loop
for i in range(epochs):
    for j, data in enumerate(train_loader):
        y_pred = model(data[0].view(-1, Sequence_Length, 1)).reshape(-1)
        loss = criterion(y_pred, data[1])
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
    if i % 50 == 0:
        print(i, "th iteration : ", loss.item())

# Test set actual vs predicted
test_set = NGTimeSeries(x_test, y_test)
test_pred = model(test_set.x.view(-1, Sequence_Length, 1)).view(-1)

# Plot predicted vs original
plt.plot(test_pred.detach().numpy(), label='Predicted')
plt.plot(test_set.y.view(-1).numpy(), label='Original')
plt.legend()
plt.show()

# Denormalize the predicted prices
y = y * (maxm - minm) + minm
y_pred = test_pred.detach().numpy() * (maxm - minm) + minm

# Plot the entire series with predictions
plt.plot(y, label='Original Prices')
plt.plot(range(len(y) - len(y_pred), len(y)), y_pred, label='Predicted Prices')
plt.legend()
plt.show()

# Prepare the input for predicting the 11th day price
last_sequence = y[-Sequence_Length:]  # Get the last 10 days of prices
last_sequence = (last_sequence - minm) / (maxm - minm)  # Normalize the last sequence
last_sequence = torch.tensor(last_sequence, dtype=torch.float32).view(1, Sequence_Length, 1)  # Reshape for the model

# Predict the price for the 11th day
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    predicted_price = model(last_sequence)

# Denormalize the predicted price
predicted_price = predicted_price.item() * (maxm - minm) + minm  # Convert back to original scale

# Print the predicted price for the 11th day
print(f"Predicted price for the 11th day: {predicted_price:.2f}")