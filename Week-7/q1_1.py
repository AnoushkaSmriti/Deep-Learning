import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import zipfile
import urllib.request
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load dataset - split into train and validation
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=validation_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Function to evaluate model
def evaluate_model(model, data_loader):
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


# Initialize model, criterion, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 regularization via weight_decay

# Training with optimizer's weight_decay
print("Training with weight_decay parameter:")
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate and print final accuracy for weight_decay method
train_accuracy = evaluate_model(model, train_loader)
val_accuracy = evaluate_model(model, val_loader)
print(f"Final results with weight_decay parameter:")
print(f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

# Reset model and optimizer for the second training approach
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # No weight decay here

# Training with manual L2 regularization
print("\nTraining with manual L2 regularization:")
l2_lambda = 0.01  # Regularization strength
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Calculate L2 regularization manually
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2) ** 2  # L2 norm squared of parameters

        loss += l2_lambda * l2_reg  # Add L2 penalty to loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate and print final accuracy for manual L2 regularization method
train_accuracy = evaluate_model(model, train_loader)
val_accuracy = evaluate_model(model, val_loader)
print(f"Final results with manual L2 regularization:")
print(f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")