import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


# Define the Custom Dropout Layer
class CustomDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.training:  # Dropout only during training
            # Bernoulli distribution: 1 - dropout_rate gives us the probability of "keeping" a neuron
            mask = torch.bernoulli(torch.full(x.shape, 1 - self.dropout_rate, device=x.device))
            x = x * mask  # Apply mask
            x = x / (1 - self.dropout_rate)  # Scale the output
        return x


# Define the CNN Model with Custom Dropout
class SimpleCNNWithCustomDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNNWithCustomDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.custom_dropout = CustomDropout(dropout_rate)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.custom_dropout(x)  # Apply custom dropout
        x = self.fc2(x)
        return x


# Define the CNN Model with Library Dropout
class SimpleCNNWithLibraryDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNNWithLibraryDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)  # Library Dropout

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply library dropout
        x = self.fc2(x)
        return x


# Define data transformations and load dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

data_dir = r'C:\Users\Reliance Digital\PycharmProjects\DeepLearning\Week-7\cats_and_dogs_filtered'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Training function
def train_model(model, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)  # Average validation loss

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss:.4f}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train with Custom Dropout
print("Training with Custom Dropout Regularization:")
model_with_custom_dropout = SimpleCNNWithCustomDropout(dropout_rate=0.5).to(device)
train_model(model_with_custom_dropout)

# Train with Library Dropout
print("\nTraining with Library Dropout Regularization:")
model_with_library_dropout = SimpleCNNWithLibraryDropout(dropout_rate=0.5).to(device)
train_model(model_with_library_dropout)
