import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Generate Fibonacci series
def generate_fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib


# Prepare the dataset
def prepare_data(fib_series):
    X = []
    y = []
    for i in range(len(fib_series) - 2):
        X.append(fib_series[i:i + 2])  # Use the last two numbers as input
        y.append(fib_series[i + 2])  # The next number as output
    return np.array(X), np.array(y)


# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the last time step
        return out


# Hyperparameters
n_fib = 20  # Number of Fibonacci numbers to generate
input_size = 1  # Each input is a single number
hidden_size = 10
output_size = 1
num_epochs = 1000
learning_rate = 0.01

# Generate Fibonacci series and prepare data
fib_series = generate_fibonacci(n_fib)
X, y = prepare_data(fib_series)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).view(-1, 2, 1)  # Shape: (samples, time_steps, features)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Shape: (samples, output_size)

# Initialize the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make a prediction for the fifth Fibonacci number
model.eval()
with torch.no_grad():
    # Use the first five Fibonacci numbers to predict the next one
    # We need to predict F(5) = 5 using F(3) = 2 and F(4) = 3
    last_two = torch.tensor(fib_series[3:5], dtype=torch.float32).view(1, 2, 1)  # F(3) and F(4)
    predicted = model(last_two)
    print(f'Predicted next Fibonacci number after F(4): {predicted.item()}')

# To explicitly show the fifth Fibonacci number
print(f'The actual fifth Fibonacci number (F(5)) is: {fib_series[5]}')