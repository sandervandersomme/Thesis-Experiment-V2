import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.models.gen_model import GenModel
from src.training.early_stopping import EarlyStopping

class TGAN(GenModel):
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    NAME = "TGAN"
        
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        self.generator = TGenerator(self.num_features, self.hidden_dim, self.num_features, self.num_layers).to(self.device)
        self.discriminator = TDiscriminator(self.num_features, self.hidden_dim, self.num_layers).to(self.device)

class TGenerator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        input_layer = nn.Linear(input_dim, hidden_dim)
        output_layer = nn.Linear(hidden_dim, output_dim)

        hidden_layers = []
        for x in range(num_layers):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())

        self.model = nn.Sequential(
            input_layer,
            nn.ReLU(),
            *hidden_layers,
            output_layer,
            nn.Sigmoid()
        )

    def forward(self, noise: torch.Tensor):
        return self.model(noise)

class TDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        input_layer = nn.Linear(input_dim, hidden_dim)
        output_layer = nn.Linear(hidden_dim, 1)

        hidden_layers = []
        for x in range(num_layers):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())

        self.model = nn.Sequential(
            input_layer,
            nn.ReLU(),
            *hidden_layers,
            output_layer,
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_TGAN(model: TGAN, train_data: torch.Tensor, epochs: int, log_loss_dir:str=None, verbose=True):
    # Setup training
    half_batch = int(model.batch_size/2)
    train_loader = DataLoader(train_data, batch_size=half_batch, shuffle=True)
    optimizer_generator = torch.optim.Adam(model.generator.parameters(), lr=model.learning_rate)
    optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=model.learning_rate)
    criterion = nn.BCELoss().to(model.device)

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    # Loss tracking
    gen_losses = []
    disc_losses = []
    val_losses = []

    # Start training loop
    for epoch in range(epochs):
        model.generator.train()
        model.discriminator.train()

        gen_loss, disc_loss = train_loss(train_loader, model, optimizer_discriminator, optimizer_generator, criterion)
        disc_losses.append(disc_loss)
        gen_losses.append(gen_loss)

        # Model validation
        model.generator.eval()
        model.discriminator.eval()
        val_loss = validation_loss(model, criterion)
        val_losses.append(val_loss.item())

        # Check if best loss has increased
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss D.: {disc_loss}, Loss G.: {gen_loss}, val loss: {val_loss}")
    
    if log_loss_dir:
        plot_losses(f"{log_loss_dir}loss.png", gen_losses, disc_losses)

    return best_val_loss
    
def train_loss(train_loader: DataLoader, model: TGAN, optim_disc: torch.optim.Adam, optim_gen: torch.optim.Adam, criterion: nn.BCELoss):
    # Initialize losses
    disc_loss = 0.0
    gen_loss = 0.0

    for _, sequences in enumerate(train_loader):
        sequences = sequences.to(model.device)

        disc_loss += discriminator_loss(model, optim_disc, criterion, sequences)
        gen_loss += generator_loss(model, optim_gen, criterion)

    avg_disc_loss = disc_loss / len(train_loader)
    avg_gen_loss = gen_loss / len(train_loader)

    return avg_gen_loss, avg_disc_loss

def discriminator_loss(model: TGAN, optimizer_discriminator: torch.optim.Adam, criterion: nn.BCELoss, events):
    # Forward pass
    noise = model.generate_noise(int(model.batch_size/2)).to(model.device)
    fake_data = model.generator(noise)
    predictions_real = model.discriminator(events)
    predictions_fake = model.discriminator(fake_data)

    # Calculate discriminator loss
    labels_fake = torch.zeros_like(predictions_fake).to(model.device)
    labels_real = torch.ones_like(predictions_real).to(model.device)

    loss_real = criterion(predictions_real, labels_real)
    loss_fake = criterion(predictions_fake, labels_fake)
    loss_discriminator = loss_real + loss_fake
    
    # Backpropogation
    optimizer_discriminator.zero_grad()
    loss_discriminator.backward()
    optimizer_discriminator.step()

    # Track loss
    return loss_discriminator.item()

def generator_loss(model: TGAN, optimizer_generator: torch.optim.Adam, criterion: nn.BCELoss):
    # Forward Generator Pass
    noise = model.generate_noise(model.batch_size).to(model.device)
    fake_data = model.generator(noise)
    predictions_fake = model.discriminator(fake_data)

    # Calculate generator loss
    labels_real = torch.ones_like(predictions_fake).to(model.device)
    loss_generator = criterion(predictions_fake, labels_real)

    # Backpropagation
    optimizer_generator.zero_grad()
    loss_generator.backward()
    optimizer_generator.step()

    # Track loss
    return loss_generator.item()

def validation_loss(model: TGAN, loss_fn: nn.BCELoss):
    """Function for calculating the validation loss based on the generator's performance in fooling the discriminator"""
    # generate fake data
    noise = model.generate_noise(model.batch_size).to(model.device)
    fake_data = model.generator(noise)

    # Run fake data through discriminator
    predictions_fake = model.discriminator(fake_data)

    # Set labels for generator data (inversed, because the loss should indicate how many the discriminator doesn't classify as real)
    labels_real = torch.ones_like(predictions_fake).to(model.device)
    return loss_fn(predictions_fake, labels_real)

def plot_losses(path, train_loss_gen, train_loss_disc):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_gen, marker='o', linestyle='-', label='train loss generator')
    plt.plot(train_loss_disc, marker='o', linestyle='-', label='train loss discriminator')

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)  # Save the figure to a file
    plt.close()