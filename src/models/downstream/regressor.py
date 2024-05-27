import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from src.models.downsteam_model import DownstreamModel
from src.utilities.early_stopping import EarlyStopping


class TimeseriesRegressor(DownstreamModel):

    __NAME__ = "Regressor"

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

        self.gru = nn.GRU(self.num_features, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

        # Send model to device
        self.to(self.device)

    def forward(self, sequences):
        output, _ = self.gru(sequences)
        output = self.fc(output)
        return output[:, -1, :] # Take classification of last time-step
    

def train_regressor(model: DownstreamModel, train_data: Dataset, val_data: Dataset, log_dir):
    writer = SummaryWriter(log_dir)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    train_loader = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=model.batch_size, shuffle=False)

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    # Loss tracking
    train_losses = []
    val_losses = []

    # Train for epochs
    for epoch in range(model.epochs):
        model.train()

        total_loss = 0
        for _, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(model.device), labels.to(model.device)


            predicted_labels = model.forward(sequences)

            loss = loss_fn(predicted_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Sum up loss for averaging later

        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (sequences, labels) in enumerate(val_loader):
                sequences, labels = sequences.to(model.device), labels.to(model.device)

                # Calculate validation loss
                outputs = model(sequences)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{model.epochs}, Average Loss: {average_loss}')
    # plot_losses(path, losses)

def plot_losses(path, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linestyle='-', label='loss regressor')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.savefig(path)


