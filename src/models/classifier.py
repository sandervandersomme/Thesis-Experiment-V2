import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from src.utilities.utils import set_device

from src.models.downsteam_model import DownstreamModel

class TimeseriesClassifier(DownstreamModel):

    __NAME__ = "ts-classifier"

    def __init__(self, shape: tuple, **hyperparams):
        super().__init__(shape, **hyperparams)
        # Architecture
        self.gru = nn.GRU(self.num_features, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        
        # Send model to device
        self.to(self.device)

    def forward(self, sequences):
        output, _ = self.gru(sequences)
        output = self.fc(output)
        return output[:, -1, :] # Take classification of last time-step
    

def train_classifier(model: TimeseriesClassifier, data: Dataset, path: str):
    loss_fn = nn.BCEWithLogitsLoss().to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    dataloader = DataLoader(data, batch_size=model.batch_size, shuffle=True)

    losses = []  # List to store loss values

    # Train for epochs
    for epoch in range(model.epochs):
        total_loss = 0
        for _, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(model.device)
            labels = labels.to(model.device)
            
            predicted_labels = model.forward(sequences)

            loss = loss_fn(predicted_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Sum up loss for averaging later

        average_loss = total_loss / len(dataloader)
        losses.append(average_loss)

        print(f'Epoch {epoch+1}/{model.epochs}, Average Loss: {average_loss}')

    plot_losses(path, losses)

def plot_losses(path, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linestyle='-', label='loss classifier')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.savefig(path)  # Save the figure to a file
    plt.show()
