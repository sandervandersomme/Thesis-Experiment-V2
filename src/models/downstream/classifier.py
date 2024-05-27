import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from src.models.downstream.downsteam_model import DownstreamModel
from src.utilities.early_stopping import EarlyStopping

class TimeseriesClassifier(DownstreamModel):

    __NAME__ = "Classifier"
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

def train_classifier(model: TimeseriesClassifier, train_data: Dataset, val_data: Dataset, log_dir):
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
        train_losses.append(avg_train_loss)

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

        # Check for early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            writer.close()
            break

        # Check if best loss has increased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Log losses to TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)

        print(f'Epoch {epoch+1}/{model.epochs}, Avg. train Loss: {avg_train_loss}, Avg. val Loss: {avg_val_loss}')

    writer.close()
    
    plot_losses(f"{model.output_path}/train-loss", train_losses)
    plot_losses(f"{model.output_path}/val-loss", val_losses)

    return best_val_loss

def plot_losses(path, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linestyle='-', label='loss classifier')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.savefig(path)  # Save the figure to a file
    plt.close()