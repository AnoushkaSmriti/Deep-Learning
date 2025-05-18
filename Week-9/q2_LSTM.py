import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define the dataset class
class NameDataset(Dataset):
    def __init__(self, data_dir):
        self.data = {}
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.getsize(file_path) == 0:  # Check if the file is empty
                print(f"Skipping empty file: {filename}")
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read names from the text file, stripping whitespace and ignoring empty lines
                names = [line.strip() for line in f if line.strip()]
                self.data[filename.split('.')[0]] = names

        self.all_letters = sorted(set(''.join(sum(self.data.values(), []))))
        self.n_letters = len(self.all_letters)
        self.lang2idx = {lang: idx for idx, lang in enumerate(self.data.keys())}
        self.idx2lang = {idx: lang for lang, idx in self.lang2idx.items()}

    def __len__(self):
        return sum(len(names) for names in self.data.values())

    def __getitem__(self, index):
        for lang, names in self.data.items():
            if index < len(names):
                return names[index], self.lang2idx[lang]
            index -= len(names)

# Define the LSTM model
class NameClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(NameClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use LSTM instead of RNN
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the last time step's output
        out = self.fc(out[:, -1, :])

        return out

# Load the dataset
dataset = NameDataset('/home/student/Downloads/data/names')
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and loss function
model = NameClassifier(input_size=dataset.n_letters, hidden_size=128, output_size=len(dataset.lang2idx))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for names, labels in train_loader:
        # Convert names to one-hot encoded tensors
        name_tensors = []
        for name in names:
            tensor = torch.zeros(len(name), dataset.n_letters)
            for j, letter in enumerate(name):
                tensor[j, dataset.all_letters.index(letter)] = 1
            name_tensors.append(tensor)

        # Pad sequences
        name_tensors = pad_sequence(name_tensors, batch_first=True)

        # Convert labels to tensor
        labels = torch.tensor(labels)

        optimizer.zero_grad()
        outputs = model(name_tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction function
def predict_language(name):
    name_tensor = torch.zeros(1, len(name), dataset.n_letters)
    for i, letter in enumerate(name):
        name_tensor[0, i, dataset.all_letters.index(letter)] = 1

    with torch.no_grad():
        output = model(name_tensor)
        _, predicted = torch.max(output, 1)
        return dataset.idx2lang[predicted.item()]

# Example usage
new_name = "Suzuki"
predicted_language = predict_language(new_name)
print(f"The predicted language of origin for '{new_name}' is '{predicted_language}'.")