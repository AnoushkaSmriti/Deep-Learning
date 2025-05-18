import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the training and validation datasets and dataloaders
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = 'cats_and_dogs_filtered'  # Specify your dataset path
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Early stopping parameters
patience = 5
best_validation_loss = float('inf')
current_patience = 0

# Initialize your model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Function to train the model with early stopping
def train_with_early_stopping(num_epochs=50):
    global best_validation_loss, current_patience
    for epoch in range(num_epochs):
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

        # Validate the model at the end of each epoch
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        validation_loss /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {validation_loss:.4f}")

        # Early stopping condition
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_patience = 0
            # Save the model when validation loss improves
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            current_patience += 1
            if current_patience >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model with early stopping
train_with_early_stopping(num_epochs=50)

# After training, you can load the best model
best_model = SimpleCNN().to(device)
best_model.load_state_dict(torch.load('best_model.pth'))