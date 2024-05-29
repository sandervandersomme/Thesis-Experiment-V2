import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from src.models.downstream.downsteam_model import DownstreamModel
from src.utilities.early_stopping import EarlyStopping

class TimeseriesClassifier(DownstreamModel):

    __NAME__ = "classifier"
    __PATH__ = f"outputs/{__NAME__}"

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)
        # Architecture
        self.gru = nn.GRU(self.num_features, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)
        
        # Send model to device
        self.to(self.device)

    def forward(self, sequences):
        output, _ = self.gru(sequences)
        output = self.fc(output)
        return output[:, -1, :] # Take classification of last time-step

def train_classifier(model: TimeseriesClassifier, train_data: Dataset, log_dir, val_data: Dataset):
    writer = SummaryWriter(log_dir)

    # Setup training
    loss_fn = nn.BCEWithLogitsLoss().to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    train_loader = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=model.batch_size, shuffle=False)

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    # Loss tracking
    train_losses = []
    val_losses = []

    # Train loop
    for epoch in range(model.epochs):
        model.train()

        loss = train_loss(train_loader, model, optimizer, loss_fn)
        train_losses.append(loss)

        model.eval()
        val_loss = validation_loss(val_loader, model, loss_fn)
        val_losses.append(val_loss)

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            writer.close()
            break

        # Check if best loss has increased
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Log losses to TensorBoard
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)

        print(f'Epoch {epoch+1}/{model.epochs}, Avg. train Loss: {loss}, Avg. val Loss: {val_loss}')

    writer.close()
    
    print(model.output_path)
    plot_losses(f"{model.output_path}-loss", train_losses, val_losses)

    return best_val_loss

def train_loss(train_loader: DataLoader, model: TimeseriesClassifier, optimizer: torch.optim.Adam, loss_fn: nn.BCEWithLogitsLoss):
    train_loss = 0
    for _, (sequences, labels) in enumerate(train_loader):
        sequences, labels = sequences.to(model.device), labels.to(model.device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences)
        loss = loss_fn(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item() # Sum up loss for averaging later

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss

def validation_loss(val_loader: DataLoader, model: TimeseriesClassifier, loss_fn: nn.BCEWithLogitsLoss):
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
    plt.plot(train_losses, marker='o', linestyle='-', label='Train loss classifier')
    plt.plot(val_losses, marker='o', linestyle='-', label='Validation loss classifier')

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)  # Save the figure to a file
    plt.close()