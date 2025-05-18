import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import PIL
import PIL.Image
from PIL import Image
import glob
import os
import numpy as np
from torchvision import models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the Gaussian noise transformation
class Gaussian(object):
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(self.mean, self.var, img.size())
        return torch.clamp(img + noise, 0, 1)  # Clamp to keep in valid range


# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, transform=None, str="train"):
        self.imgs_path = "cats_and_dogs_filtered/" + str + "/"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = PIL.Image.open(img_path).convert("RGB")
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id, dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, class_id


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.25)  # Add dropout for regularization

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x


# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * correct / total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies


# Define the transformations for data augmentation
preprocess_augment = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(45),
    T.ToTensor(),
    Gaussian(0, 0.05),  # Reduced variance for better stability
])

# Define the transformations without data augmentation
preprocess_no_augment = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


# Function to visualize augmented images
def display_augmented_images(dataset, num_images=3):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        img, _ = dataset[i]
        plt.subplot(1, num_images, i + 1)
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"Augmented Image {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Experiment 1: Train with data augmentation
print("Experiment 1: Training with Data Augmentation")
dataset_aug = MyDataset(transform=preprocess_augment, str="train")
train_size = int(0.8 * len(dataset_aug))
val_size = len(dataset_aug) - train_size
train_dataset_aug, val_dataset_aug = random_split(dataset_aug, [train_size, val_size])

# Display augmented images
print("Sample augmented images:")
display_augmented_images(dataset_aug)

# Create data loaders for augmented data
train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True)
val_loader_aug = DataLoader(val_dataset_aug, batch_size=32, shuffle=False)

# Initialize model, criterion, and optimizer
model_aug = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_aug = optim.Adam(model_aug.parameters(), lr=0.001)

# Train the model with augmented data
aug_train_losses, aug_val_losses, aug_train_accs, aug_val_accs = train_model(
    model_aug, train_loader_aug, val_loader_aug, criterion, optimizer_aug
)

# Experiment 2: Train without data augmentation
print("\nExperiment 2: Training without Data Augmentation")
dataset_no_aug = MyDataset(transform=preprocess_no_augment, str="train")
train_dataset_no_aug, val_dataset_no_aug = random_split(dataset_no_aug, [train_size, val_size])

# Create data loaders for non-augmented data
train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=32, shuffle=True)
val_loader_no_aug = DataLoader(val_dataset_no_aug, batch_size=32, shuffle=False)

# Initialize model, criterion, and optimizer
model_no_aug = SimpleCNN().to(device)
optimizer_no_aug = optim.Adam(model_no_aug.parameters(), lr=0.001)

# Train the model without augmented data
no_aug_train_losses, no_aug_val_losses, no_aug_train_accs, no_aug_val_accs = train_model(
    model_no_aug, train_loader_no_aug, val_loader_no_aug, criterion, optimizer_no_aug
)

# Plot results
plt.figure(figsize=(15, 10))

# Plot training and validation loss
plt.subplot(2, 2, 1)
plt.plot(aug_train_losses, label='With Augmentation - Train')
plt.plot(aug_val_losses, label='With Augmentation - Val')
plt.plot(no_aug_train_losses, label='Without Augmentation - Train')
plt.plot(no_aug_val_losses, label='Without Augmentation - Val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(2, 2, 2)
plt.plot(aug_train_accs, label='With Augmentation - Train')
plt.plot(aug_val_accs, label='With Augmentation - Val')
plt.plot(no_aug_train_accs, label='Without Augmentation - Train')
plt.plot(no_aug_val_accs, label='Without Augmentation - Val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot validation accuracy comparison
plt.subplot(2, 2, 3)
plt.plot(aug_val_accs, 'g-', label='With Augmentation')
plt.plot(no_aug_val_accs, 'r-', label='Without Augmentation')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy Comparison')
plt.legend()

# Plot validation loss comparison
plt.subplot(2, 2, 4)
plt.plot(aug_val_losses, 'g-', label='With Augmentation')
plt.plot(no_aug_val_losses, 'r-', label='Without Augmentation')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()

plt.tight_layout()
plt.show()

# Print final results
print("\nFinal Results:")
print(f"With Data Augmentation - Final Validation Accuracy: {aug_val_accs[-1]:.2f}%")
print(f"Without Data Augmentation - Final Validation Accuracy: {no_aug_val_accs[-1]:.2f}%")

# Calculate the gap between train and validation accuracy (to measure overfitting)
aug_gap = aug_train_accs[-1] - aug_val_accs[-1]
no_aug_gap = no_aug_train_accs[-1] - no_aug_val_accs[-1]

print(f"\nOverfitting Analysis:")
print(f"With Data Augmentation - Train/Val Accuracy Gap: {aug_gap:.2f}%")
print(f"Without Data Augmentation - Train/Val Accuracy Gap: {no_aug_gap:.2f}%")