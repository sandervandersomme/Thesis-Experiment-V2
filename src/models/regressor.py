import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from src.utilities.utils import set_device

class TimeseriesRegressor(nn.Module):

    __MODEL__ = "ts-regressor"

    # TODO: Adapt architecture
    def __init__(self, device: str, seed: int, **hyperparams):
        super().__init__()

        # torch settings
        if "device" in hyperparams: self.device = hyperparams["device"]
        else: self.device = set_device()

        if seed:
            torch.manual_seed(seed)

        # Architecture
        self.input_dim = hyperparams["input_dim"]
        self.hidden_dim = hyperparams["hidden_dim"]
        self.num_layers = hyperparams["num_layers"]

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

        # Send model to device
        self.to(self.device)

        # Training parameters
        self.batch_size = hyperparams["batch_size"]
        self.epochs = hyperparams["epochs"]
        self.learning_rate = hyperparams['learning_rate']

    def forward(self, sequences):
        output, _ = self.gru(sequences)
        output = self.fc(output)
        return output[:, -1, :] # Take classification of last time-step
    

def train_regressor(self, model: TimeseriesRegressor, data: Dataset, **hyperparams):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    dataloader = DataLoader(data, batch_size=hyperparams["batch_size"], shuffle=True)

    losses = []  # List to store loss values

    # Train for epochs
    for epoch in range(hyperparams["epochs"]):
        total_loss = 0
        for _, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(model.device)
            labels = labels.to(model.device)

            predicted_labels = model.forward(sequences)

            loss = loss_fn(predicted_labels, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()  # Sum up loss for averaging later

        average_loss = total_loss / len(dataloader)
        losses.append(average_loss)

        print(f'Epoch {epoch+1}/{hyperparams["epochs"]}, Average Loss: {average_loss}')
    plot_losses(losses)

def plot_losses(path, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linestyle='-', label='loss regressor')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.savefig(path)


