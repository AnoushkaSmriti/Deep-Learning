import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)

model = LogisticRegressionModel()
loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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

plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.grid()
plt.show()

test_input = torch.tensor([[3]], dtype=torch.float32)
predicted_prob = model(test_input).item()
predicted_class = 1 if predicted_prob >= 0.5 else 0

print(f"Predicted probability for X=3: {predicted_prob:.4f}")
print(f"Predicted class for X=3: {predicted_class}")
