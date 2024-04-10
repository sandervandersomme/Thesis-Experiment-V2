import torch
import torch.nn as nn
import numpy as np
from GAN import Generator, Discriminator
from utils import generate_noise, visualise

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class TimeGAN():
    """
    Time-series GAN consisting of a generator, discriminator, embedder and recovery function.
    """
    
    __MODEL__ = "TimeGAN"

    def __init__(self, data: torch.Tensor, device: str, hidden_dim: int, latent_dim: int, batch_size: int, seed: int = None):
        self.device = device
        self.data = data
        self.data_loader = DataLoader(data, batch_size=1, shuffle=True)

        # set seed for reproducibility
        if seed:
            torch.manual_seed(seed)

        # set parameters
        _, self.seq_length, self.num_of_features = self.data.shape
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Create architecture
        self.generator = Generator(self.num_of_features, self.hidden_dim, latent_dim)
        self.discriminator = Discriminator(latent_dim, self.hidden_dim)
        self.supervisor = Supervisor(latent_dim, hidden_dim, latent_dim)
        self.embedder = Embedder(self.num_of_features, hidden_dim, latent_dim)
        self.recovery = Recovery(latent_dim, hidden_dim, self.num_of_features)

    def generate_data(self, num_samples: int, data_path: str, epochs: int):
        noise = generate_noise(num_samples, self.seq_length, self.num_of_features)
        with torch.no_grad():
            generated_embeddings = self.generator(noise)
            supervised_embeddings = self.supervisor(generated_embeddings)
            generated_data = self.recovery(supervised_embeddings)
            np.save(f"output/syndata/{self.__MODEL__}/{data_path}_{epochs}", generated_data.numpy())
            return generated_data    
        
    def train(self, epochs: int, learning_rate: float):

        # Initialising loss functions
        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()

        # Initialising optimizers
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        embedder_optimizer = torch.optim.Adam(self.embedder.parameters(), lr=learning_rate)
        recovery_optimizer = torch.optim.Adam(self.recovery.parameters(), lr=learning_rate)
        supervisor_optimizer = torch.optim.Adam(self.supervisor.parameters(), lr=learning_rate)

        print('Start Training Phase 1: Minimize reconstruction loss')

        # Track losses
        losses_reconstruction = torch.zeros(epochs)
        losses_supervised = torch.zeros(epochs)
        losses_generator = torch.zeros(epochs)
        losses_embedder = torch.zeros(epochs)
        losses_discriminator = torch.zeros(epochs)

        # Training phase 1: Minimize reconstruction loss
        for epoch in range(epochs):
            for _, real_sequences in enumerate(self.data_loader):

                # Minimize reconstruction loss
                embedder_optimizer.zero_grad()
                recovery_optimizer.zero_grad()

                embeddings = self.embedder(real_sequences)
                reconstructions = self.recovery(embeddings)

                reconstruction_loss = mse_loss(reconstructions, real_sequences)
                reconstruction_loss.backward()

                embedder_optimizer.step()
                recovery_optimizer.step()
            
            losses_reconstruction[epoch] = reconstruction_loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Reconstruction Loss: {reconstruction_loss.item()}")
      
        print('Start Training Phase 2: Minimize unsupervised loss')
        visualise(epochs, losses_reconstruction, "Epochs", "Reconstruction loss", "Reconstruction loss phase 1", "TimeGAN reconstruction")

        # Training phase 2: Minimize supervised loss
        for epoch in range(epochs):
            for _, real_sequences in enumerate(self.data_loader):

                # Train supervisor
                supervisor_optimizer.zero_grad()
                generator_optimizer.zero_grad()

                embeddings = self.embedder(real_sequences)
                supervised_embeddings = self.supervisor(embeddings)

                supervised_loss = mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])
                supervised_loss.backward()

                generator_optimizer.step()
                supervisor_optimizer.step()
            
            losses_supervised[epoch] = supervised_loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Supervised Loss: {supervised_loss.item()}")

        visualise(epochs, losses_supervised, "Epochs", "Supervised loss", "'Supervised loss phase 2'", "TimeGAN Supervised loss")
        print('Start Training Phase 3: Joint training')

        # Training phase 3: Joint training
        for epoch in range(epochs):
            # Train generator and supervisor twice as much as discriminator
            for _ in range(2):
                for _, real_sequences in enumerate(self.data_loader):
                    # Reset the gradients
                    generator_optimizer.zero_grad()
                    supervisor_optimizer.zero_grad()
                    discriminator_optimizer.zero_grad()
                    embedder_optimizer.zero_grad()

                    # Calculate unsupervised loss: noise --> generator --> supervisor --> discriminator --> loss
                    noise = generate_noise(self.batch_size, self.seq_length, self.num_of_features)
                    generated_embeddings = self.generator(noise)
                    supervised_embeddings = self.supervisor(generated_embeddings)
                    predictions_fake = self.discriminator(supervised_embeddings)
                    unsupervised_loss = bce_loss(predictions_fake, torch.ones_like(predictions_fake))

                    #TODO: Add moments loss

                    # calculate supervised loss: real data --> embedder --> supervisor --> loss
                    embeddings = self.embedder(real_sequences)
                    supervised_embeddings = self.supervisor(embeddings)
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
                    embeddings = self.embedder(real_sequences)
                    reconstructions = self.recovery(embeddings)
                    reconstruction_loss = mse_loss(reconstructions, real_sequences)

                    # Calculate supervised_loss: real data --> embedder --> supervisor --> loss
                    supervised_embeddings = self.supervisor(embeddings)
                    supervised_loss = mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])

                    # Combine losses and calculate gradients
                    embedder_loss = supervised_loss + reconstruction_loss
                    embedder_loss.backward()

                    # Only update embedder and recovery networks
                    embedder_optimizer.step()
                    recovery_optimizer.step()

            # Train discriminator
            for batch_dix, real_sequences in enumerate(self.data_loader):
                # Reset the gradients of the optimizers
                discriminator_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                supervisor_optimizer.zero_grad()
                embedder_optimizer.zero_grad()

                # Forward pass
                noise = generate_noise(self.batch_size, self.seq_length, self.num_of_features)
                fake_embeddings = self.generator(noise)
                supervised_embeddings = self.supervisor(fake_embeddings)
                real_embeddings = self.embedder(real_sequences)

                predictions_fake = self.discriminator(fake_embeddings)
                predictions_real = self.discriminator(real_embeddings)
                predictions_supervised = self.discriminator(supervised_embeddings)

                # Calculate discriminator loss
                loss_fake = bce_loss(predictions_fake, torch.zeros_like(predictions_fake))
                loss_supervised = bce_loss(predictions_supervised, torch.zeros_like(predictions_supervised))
                loss_real = bce_loss(predictions_real, torch.ones_like(predictions_fake))
                discriminator_loss = loss_fake + loss_real + loss_supervised
                discriminator_loss.backward()

                # Only update the discriminator
                discriminator_optimizer.step()

            # Track joint losses
            losses_discriminator[epoch] = discriminator_loss.item()
            losses_generator[epoch] = generator_loss.item()
            losses_embedder[epoch] = embedder_loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Generator loss: {generator_loss.item()}, Embedder loss: {embedder_loss.item()}, Discriminator Loss: {discriminator_loss.item()}")

        visualise(epochs, [losses_discriminator, losses_generator, losses_embedder], "Epochs", "Loss", ["Discriminator loss", "Generator loss", "Embedder loss"], "TimeGAN Joint Training")




class Embedder(nn.Module):
    """
    A GRU based embedder that takes in real sequences and converts it into a latent representation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Embedder, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, real_sequences):
        output, _ = self.rnn(real_sequences)
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