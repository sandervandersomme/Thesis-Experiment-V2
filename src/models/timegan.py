import torch
import torch.nn as nn

from src.models.gan import Generator, Discriminator
from src.models.gen_model import GenModel

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Embedder(nn.Module):
    """
    A GRU based embedder that takes in real sequences and converts it into a latent representation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Embedder, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, sequences):
        output, _ = self.rnn(sequences)
        return self.output_layer(output)
    
class Recovery(nn.Module):
    """
    A GRU based recovery network that takes in a latent representation and returns a reconstructed sample
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Recovery, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, latent_vector):
        rnn_output, _ = self.rnn(latent_vector)
        return self.output_layer(rnn_output)

class Supervisor(nn.Module):
    """
    A GRU based supervisor that takes in latent representations generated by the generator and returns the sequence generated from the latent representation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Supervisor, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, latent_representations):
        output, _ = self.rnn(latent_representations)
        return self.output_layer(output)

class TimeGAN(GenModel):
    """
    Time-series GAN consisting of a generator, discriminator, embedder and recovery function.
    """
    
    __NAME__ = "TimeGAN"

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

        # Set hyperparams
        self.latent_dim = hyperparams["latent_dim"]

        # Create architecture
        self.generator = Generator(self.num_features, self.hidden_dim, self.latent_dim).to(self.device)
        self.discriminator = Discriminator(self.latent_dim, self.hidden_dim).to(self.device)
        self.supervisor = Supervisor(self.latent_dim, self.hidden_dim, self.latent_dim).to(self.device)
        self.embedder = Embedder(self.num_features, self.hidden_dim, self.latent_dim).to(self.device)
        self.recovery = Recovery(self.latent_dim, self.hidden_dim, self.num_features).to(self.device)

    def generate_data(self, num_samples: int):
        noise = self.generate_noise(num_samples)
        with torch.no_grad():
            generated_embeddings = self.generator(noise)
            supervised_embeddings = self.supervisor(generated_embeddings)
            generated_data = self.recovery(supervised_embeddings)
            return generated_data    
        
TimeGAN_params = {
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 10,
    "hidden_dim": 10,
    "num_layers": 1,
    "latent_dim": 10
}


def train_TimeGAN(model, data: torch.Tensor, path: str):
    data_loader = DataLoader(data, model.batch_size, shuffle=True)

    # Initialising loss functions
    mse_loss = torch.nn.MSELoss().to(model.device)
    bce_loss = torch.nn.BCELoss().to(model.device)

    # Initialising optimizers
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=model.learning_rate)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=model.learning_rate)
    embedder_optimizer = torch.optim.Adam(model.embedder.parameters(), lr=model.learning_rate)
    recovery_optimizer = torch.optim.Adam(model.recovery.parameters(), lr=model.learning_rate)
    supervisor_optimizer = torch.optim.Adam(model.supervisor.parameters(), lr=model.learning_rate)

    # Train
    losses_reconstruction = phase_1(model, data_loader, embedder_optimizer, recovery_optimizer, mse_loss)
    losses_supervised = phase_2(model, data_loader, supervisor_optimizer, generator_optimizer, mse_loss)
    losses_discriminator, losses_generator, losses_embedder = phase_3(model, data_loader, supervisor_optimizer, generator_optimizer, discriminator_optimizer, embedder_optimizer, recovery_optimizer, bce_loss, mse_loss)

    visualise_phase1(path, losses_reconstruction)
    visualise_phase2(path, losses_supervised)
    visualise_phase3(path, losses_discriminator, losses_generator, losses_embedder)

def phase_1(model, dataloader: DataLoader, embedder_optimizer: torch.optim.Adam, recovery_optimizer: torch.optim.Adam, mse_loss: torch.nn.MSELoss):
        # Training phase 1: Minimize reconstruction loss
    print('Start Training Phase 1: Minimize reconstruction loss')

    losses_reconstruction = torch.zeros(model.epochs)

    for epoch in range(model.epochs):
        for _, sequences in enumerate(dataloader):
            sequences = sequences.to(model.device)


            # Minimize reconstruction loss
            embedder_optimizer.zero_grad()
            recovery_optimizer.zero_grad()

            embeddings = model.embedder(sequences)
            reconstructions = model.recovery(embeddings)

            reconstruction_loss = mse_loss(reconstructions, sequences)
            reconstruction_loss.backward()

            embedder_optimizer.step()
            recovery_optimizer.step()
        
        losses_reconstruction[epoch] = reconstruction_loss.item()
        print(f"Epoch {epoch+1}/{model.epochs}, Reconstruction Loss: {reconstruction_loss.item()}")
    print("Training phase 1 complete!")

    return losses_reconstruction

def phase_2(model, dataloader: DataLoader, supervisor_optimizer: torch.optim.Adam, generator_optimizer: torch.optim.Adam, mse_loss: torch.nn.MSELoss):
    # Training phase 2: Minimize supervised loss
    print('Start Training Phase 2: Minimize unsupervised loss')

    losses_supervised = torch.zeros(model.epochs)

    for epoch in range(model.epochs):
        for _, sequences in enumerate(dataloader):
            sequences = sequences.to(model.device)

            # Train supervisor
            supervisor_optimizer.zero_grad()
            generator_optimizer.zero_grad()

            embeddings = model.embedder(sequences)
            supervised_embeddings = model.supervisor(embeddings)

            supervised_loss = mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])
            supervised_loss.backward()

            generator_optimizer.step()
            supervisor_optimizer.step()
        
        losses_supervised[epoch] = supervised_loss.item()
        print(f"Epoch {epoch+1}/{model.epochs}, Supervised Loss: {supervised_loss.item()}")

    return losses_supervised
    
def phase_3(model, dataloader: DataLoader, supervisor_optimizer: torch.optim.Adam, generator_optimizer: torch.optim.Adam, discriminator_optimizer: torch.optim.Adam, embedder_optimizer: torch.optim.Adam, recovery_optimizer: torch.optim.Adam, bce_loss: torch.nn.BCELoss, mse_loss: torch.nn.BCELoss):
    # Training phase 3: Joint training
    print('Start Training Phase 3: Joint training')

    # Track losses
    losses_generator = torch.zeros(model.epochs)
    losses_embedder = torch.zeros(model.epochs)
    losses_discriminator = torch.zeros(model.epochs)

    for epoch in range(model.epochs):
        # Train generator and supervisor twice as much as discriminator
        for _ in range(2):
            for _, sequences in enumerate(dataloader):
                sequences = sequences.to(model.device)

                # Reset the gradients
                generator_optimizer.zero_grad()
                supervisor_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()
                embedder_optimizer.zero_grad()

                # Calculate unsupervised loss: noise --> generator --> supervisor --> discriminator --> loss
                noise = model.generate_noise(model.batch_size)
                generated_embeddings = model.generator(noise)
                supervised_embeddings = model.supervisor(generated_embeddings)
                predictions_fake = model.discriminator(supervised_embeddings)
                unsupervised_loss = bce_loss(predictions_fake, torch.ones_like(predictions_fake).to(model.device))

                #TODO: Add moments loss

                # calculate supervised loss: real data --> embedder --> supervisor --> loss
                embeddings = model.embedder(sequences)
                supervised_embeddings = model.supervisor(embeddings)
                supervised_loss = mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])

                # Combine losses and calculate gradients
                generator_loss = supervised_loss + unsupervised_loss
                generator_loss.backward()

                # Update generator and supervisor
                generator_optimizer.step()
                supervisor_optimizer.step()

                # Reset gradients
                embedder_optimizer.zero_grad()
                recovery_optimizer.zero_grad()
                supervisor_optimizer.zero_grad()

                # Calculate reconstruction loss: real data --> embedder --> recovery --> loss
                embeddings = model.embedder(sequences)
                reconstructions = model.recovery(embeddings)
                reconstruction_loss = mse_loss(reconstructions, sequences)

                # Calculate supervised_loss: real data --> embedder --> supervisor --> loss
                supervised_embeddings = model.supervisor(embeddings)
                supervised_loss = mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])

                # Combine losses and calculate gradients
                embedder_loss = supervised_loss + reconstruction_loss
                embedder_loss.backward()

                # Only update embedder and recovery networks
                embedder_optimizer.step()
                recovery_optimizer.step()

        # Train discriminator
        for _, sequences in enumerate(dataloader):
            sequences = sequences.to(model.device)

            # Reset the gradients of the optimizers
            generator_optimizer.zero_grad()
            supervisor_optimizer.zero_grad()
            embedder_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # Forward pass
            noise = model.generate_noise(len(sequences))
            fake_embeddings = model.generator(noise)
            supervised_embeddings = model.supervisor(fake_embeddings)
            real_embeddings = model.embedder(sequences)

            predictions_fake = model.discriminator(fake_embeddings)
            predictions_real = model.discriminator(real_embeddings)
            predictions_supervised = model.discriminator(supervised_embeddings)

            # Calculate discriminator loss
            loss_fake = bce_loss(predictions_fake, torch.zeros_like(predictions_fake).to(model.device))
            loss_supervised = bce_loss(predictions_supervised, torch.zeros_like(predictions_supervised))
            loss_real = bce_loss(predictions_real, torch.ones_like(predictions_fake).to(model.device))

            discriminator_loss = loss_fake + loss_real + loss_supervised
            discriminator_loss.backward()
            discriminator_optimizer.step()

        # Track joint losses
        losses_discriminator[epoch] = discriminator_loss.item()
        losses_generator[epoch] = generator_loss.item()
        losses_embedder[epoch] = embedder_loss.item()
        print(f"Epoch {epoch+1}/{model.epochs}, Generator loss: {generator_loss.item()}, Embedder loss: {embedder_loss.item()}, Discriminator Loss: {discriminator_loss.item()}")

    return losses_discriminator, losses_generator, losses_embedder

def visualise_phase1(path: str, reconstruction_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(reconstruction_loss.numpy(), marker='o', linestyle='-', label = "Reconstruction Loss")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{path}/reconstruction")  # Save the figure to a file
    plt.clf()

def visualise_phase2(path: str, supervised_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(supervised_loss.numpy(), marker='o', linestyle='-', label = "Supervised Loss")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{path}/supervised")  # Save the figure to a file
    plt.clf()

def visualise_phase3(path: str, losses_generator, losses_discriminator, losses_embedder):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_generator.numpy(), marker='o', linestyle='-', label = "Generator Loss")
    plt.plot(losses_discriminator.numpy(), marker='o', linestyle='-', label = "Discriminator Loss")
    plt.plot(losses_embedder.numpy(), marker='o', linestyle='-', label = "Embedder Loss")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{path}/Joint")  # Save the figure to a file
    plt.clf()

