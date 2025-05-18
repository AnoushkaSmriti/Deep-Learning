import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import zipfile
import urllib.request
import numpy as np

# Download and extract dataset if not already present
data_dir = 'cats_and_dogs_filtered'
if not os.path.exists(data_dir):
    print("Downloading Cats and Dogs dataset...")
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    dataset_path = "cats_and_dogs_filtered.zip"

    urllib.request.urlretrieve(dataset_url, dataset_path)

    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall()

    print(f"Dataset extracted to {data_dir}")


# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjusted for 256x256 image size
        self.fc2 = nn.Linear(512, 2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Correct flatten size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load dataset
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Function to evaluate model
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, criterion, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with L1 Regularization
l1_lambda = 0.01
epochs = 10
print("Training with L1 regularization:")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # L1 regularization
        l1_reg = 0
        for param in model.parameters():
            l1_reg += torch.sum(torch.abs(param))  # L1 norm

        loss += l1_lambda * l1_reg  # Add L1 penalty to the loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate and print final accuracy
train_accuracy = evaluate_model(model, train_loader, device)
val_accuracy = evaluate_model(model, val_loader, device)
print(f"\nFinal results with L1 regularization (lambda={l1_lambda}):")
print(f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

# Let's examine the sparsity of the model parameters (a key feature of L1 regularization)
total_params = 0
zero_params = 0
near_zero_params = 0
threshold = 1e-3

for name, param in model.named_parameters():
    param_data = param.data.cpu().numpy().flatten()
    total_params += param_data.size
    zero_params += (param_data == 0).sum()
    near_zero_params += ((param_data != 0) & (np.abs(param_data) < threshold)).sum()

print(f"\nParameter sparsity analysis:")
print(f"Total parameters: {total_params}")
print(f"Zero parameters: {zero_params} ({zero_params / total_params * 100:.2f}%)")
print(f"Near-zero parameters (< {threshold}): {near_zero_params} ({near_zero_params / total_params * 100:.2f}%)")
print(f"Total sparsity: {(zero_params + near_zero_params) / total_params * 100:.2f}%")