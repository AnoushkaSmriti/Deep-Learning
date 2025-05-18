import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature and one output

    def forward(self, x):
        return self.linear(x)

x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9,
                  18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2,
                  17.4, 19.5, 19.7, 21.2], dtype=torch.float32).view(-1, 1)  # Reshape for nn.Linear
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9,
                  15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1,
                  16.8, 15.2, 17.0, 17.2, 18.6], dtype=torch.float32).view(-1, 1)

model = LinearRegressionModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 10
losses = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients

    y_pred = model(x)  # Forward pass
    loss = loss_fn(y_pred, y)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    losses.append(loss.item())  # Store loss

    with torch.no_grad():  # No need to track gradients for printing

        params_and_grads = []

        for name, param in model.named_parameters():
            params_and_grads.append(f"{name}: {param.data.numpy()}, Gradient: {param.grad.numpy()}")

        print(f'Epoch {epoch + 1}/{epochs}, ' + ', '.join(params_and_grads))

plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.grid()
plt.show()
