import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder maps 784 -> 100 -> 10
        self.Encoder = nn.Sequential(
            nn.Linear(784, 100, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 10, bias=True),
            nn.Sigmoid(),
        )
        # Compute latent mean and log variance from the encoder output (10-dimensional)
        self.mean = nn.Linear(10, 10, bias=True)
        self.log_var = nn.Linear(10, 10, bias=True)

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
        enc = self.Encoder(x)
        mean = self.mean(enc)
        log_var = self.log_var(enc)
        # Reparameterization trick
        z = mean + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        y = self.Decoder(z)
        return mean, log_var, y.view(-1, 1, 28, 28)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

# MNIST DataLoader
batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set device, instantiate model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = loss_function

def train_one_epoch(epoch_index):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, _ = data
        inputs = inputs.to(device)
        # Flatten the input images to a 784-dimensional vector
        inputs_flat = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        mean, log_var, outputs = model(inputs_flat)
        loss = loss_fn(inputs, outputs, mean, log_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (len(train_data_loader) * batch_size)

def generate_digit():
    model.eval()
    # Sample from a standard normal latent space
    mean = torch.zeros((1, 10))
    var = torch.ones((1, 10))
    z_sample = mean + var * torch.randn_like(var)
    x_decoded = model.Decoder(z_sample.to(device))
    digit = x_decoded.detach().cpu().reshape(28, 28)
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.show()

# Example training loop
num_epochs = 10
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(epoch)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")


# Generate a synthesized digit after training
generate_digit()
generate_digit()

