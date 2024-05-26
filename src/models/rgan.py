import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.gan import Generator, Discriminator
import matplotlib.pyplot as plt

from src.models.gen_model import GenModel

class RGAN(GenModel):
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    __MODEL__ = "RGAN"
        
    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

        # Create architecture
        self.generator = Generator(self.num_features, self.hidden_dim, self.num_features).to(self.device)
        self.discriminator = Discriminator(self.num_features, self.hidden_dim).to(self.device)

RGAN_params = {
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 200,
    "hidden_dim": 10,
    "num_layers": 1
}

def train_RGAN(model: GenModel, data: torch.Tensor, path):
    half_batch = int(model.batch_size/2)
    data_loader = DataLoader(data, batch_size=half_batch, shuffle=True)

    # Set optimizers
    optimizer_generator = torch.optim.Adam(model.generator.parameters(), lr=model.learning_rate)
    optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=model.learning_rate)
    
    # Set loss function
    criterion = nn.BCELoss().to(model.device)

    # Track losses per epochs
    losses_generator = torch.zeros(model.epochs)
    losses_discriminator = torch.zeros(model.epochs)

    for epoch_id in range(model.epochs):
        for _, sequences in enumerate(data_loader):
            sequences = sequences.to(model.device)

            losses_discriminator[epoch_id] = train_discriminator(model, optimizer_discriminator, criterion, sequences)
            losses_generator[epoch_id] = train_generator(model, optimizer_generator, criterion)

        print(f"Epoch {epoch_id+1}/{model.epochs}, Loss D.: {losses_discriminator[epoch_id].item()}, Loss G.: {losses_generator[epoch_id].item()}")
    
    visualise(losses_generator, losses_discriminator, path)

def train_generator(model: RGAN, optimizer_generator: torch.optim.Adam, criterion: nn.BCELoss):
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

def train_discriminator(model: GenModel, optimizer_discriminator: torch.optim.Adam, criterion: nn.BCELoss, sequences):
    # Forward pass
    noise = model.generate_noise(int(model.batch_size/2)).to(model.device)
    fake_data = model.generator(noise)
    predictions_real = model.discriminator(sequences)
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
  
def visualise(losses_generator, losses_discriminator, path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_generator, marker='o', linestyle='-', label='loss generator')
    plt.plot(losses_discriminator, marker='o', linestyle='-', label='loss discriminator')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)  # Save the figure to a file
    plt.clf()
    


