import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from src.models.downsteam_model import DownstreamModel
from src.training.early_stopping import EarlyStopping


class TimeseriesRegressor(DownstreamModel):

    NAME = "regressor"

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
    
def train_regressor(model: DownstreamModel, train_data: Dataset, val_data: Dataset, epochs: int, plot_path:str=None, verbose=True):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    train_loader = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
    if val_data:
        val_loader = DataLoader(val_data, batch_size=model.batch_size, shuffle=False)

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    # Loss tracking
    train_losses = []
    val_losses = []

    # Start training loop
    for epoch in range(epochs):
        # Train model
        model.train()
        loss = train_loss(train_loader, model, optimizer, loss_fn)
        train_losses.append(loss.item())

        # Validate model
        model.eval()
        val_loss = validation_loss(val_loader, model, loss_fn)
        val_losses.append(val_loss)

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

        # Check if best loss has increased (for hyperparameter optimization)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Avg. train Loss: {loss}, Avg. val Loss: {val_loss}')
    
    # Visualise losses over epochs
    if plot_path:
        plot_losses(f"{plot_path}loss.png", train_losses, val_losses)

    return best_val_loss

def train_loss(train_loader: DataLoader, model: TimeseriesRegressor, optimizer: torch.optim.Adam, loss_fn: nn.MSELoss):
    loss = 0
    for _, (sequences, labels) in enumerate(train_loader):
        sequences, labels = sequences.to(model.device), labels.to(model.device)

        # Forward pass and loss calculation
        predicted_labels = model.forward(sequences)
        loss = loss_fn(predicted_labels, labels)

        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss += loss.item()  # Sum up loss for averaging later

    average_loss = loss / len(train_loader)
    return average_loss

def validation_loss(val_loader: DataLoader, model:TimeseriesRegressor, loss_fn: nn.MSELoss):
    val_loss = 0
    with torch.no_grad():
        for _, (sequences, labels) in enumerate(val_loader):
            
            sequences, labels = sequences.to(model.device), labels.to(model.device)

            # Calculate validation loss
            outputs = model(sequences)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

def plot_losses(path, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o', linestyle='-', label='train loss regressor')
    plt.plot(val_losses, marker='o', linestyle='-', label='validation loss regressor')

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)
    plt.close()


