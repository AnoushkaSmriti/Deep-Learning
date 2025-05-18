import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder maps 784 -> 100 -> 10 (latent representation)
        self.Encoder = nn.Sequential(
            nn.Linear(784, 100, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 10, bias=True),
            nn.Sigmoid(),  # Ensures latent values are between 0 and 1
        )
        # Decoder reconstructs the image from the latent vector
        self.Decoder = nn.Sequential(
            nn.BatchNorm1d(10),
            nn.Linear(10, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, 784, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.Encoder(x)
        x_reconstructed = self.Decoder(latent)
        return latent, x_reconstructed.view(-1, 1, 28, 28)

def loss_function(x, x_hat):
    # Reconstruction loss only (binary cross entropy)
    return F.binary_cross_entropy(x_hat, x, reduction='sum')

# MNIST DataLoader
batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set device, instantiate model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(epoch_index):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, _ = data
        inputs = inputs.to(device)
        # Flatten the input images to a 784-dimensional vector
        inputs_flat = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        latent, outputs = model(inputs_flat)
        loss = loss_function(inputs, outputs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / (len(train_data_loader) * batch_size)


def generate_digit():
    model.eval()
    # Sample a random latent vector from a uniform distribution in [0,1]
    latent_sample = torch.rand((1, 10))
    x_decoded = model.Decoder(latent_sample.to(device))
    digit = x_decoded.detach().cpu().reshape(28, 28)
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

# Example training loop
num_epochs = 10
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(epoch)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")


# Generate a reconstructed digit using a random latent vector
generate_digit()
generate_digit()
