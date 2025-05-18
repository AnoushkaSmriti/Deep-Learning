import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import requests
import zipfile
from io import BytesIO

def download_dataset(url, extract_to='.'):
    # Create data directory
    os.makedirs(extract_to, exist_ok=True)

    # Download and extract dataset
    print("Downloading dataset...")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)
    print("Dataset downloaded and extracted")

# Define transformations
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Download dataset
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
download_dataset(url, extract_to='.')

# Set data directory
data_dir = 'cats_and_dogs_filtered'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform['train'])
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform['val'])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the pre-trained model
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# Freeze the parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Training epochs
epochs = 5

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    best_acc = 0.0
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

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Final Accuracy on the validation set: {accuracy * 100:.2f}%")

def visualize_predictions(model, val_loader):
    model.eval()
    data_iter = iter(val_loader)
    inputs, labels = next(data_iter)
    outputs = model(inputs.to(device))
    _, predicted = torch.max(outputs, 1)

    # Denormalize the images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    inputs = inputs * std + mean  # Denormalize
    inputs = torch.clamp(inputs, 0, 1)  # Ensure values are in [0, 1]

    fig, axes = plt.subplots(1, 5, figsize=(15, 15))
    for i in range(5):
        axes[i].imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f"Pred: {'Dog' if predicted[i] == 1 else 'Cat'}, True: {'Dog' if labels[i] == 1 else 'Cat'}")
        axes[i].axis('off')

    plt.show()

visualize_predictions(model, val_loader)