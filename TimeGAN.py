import torch
import torch.nn as nn
import numpy as np
from GAN import Generator, Discriminator
from utils import generate_noise

from parameters import hyperparams

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TimeGAN():
    """
    Time-series GAN consisting of a generator, discriminator, embedder and recovery function.
    """
    
    __MODEL__ = "TimeGAN"

    def __init__(self, shape: tuple, device: str, seed: int = None):
        self.device = device

        # set seed for reproducibility
        if seed:
            torch.manual_seed(seed)

        # set parameters
        _, self.seq_length, self.num_of_features = shape

        # Create architecture
        self.generator = Generator(self.num_of_features, hyperparams["hidden_dim"], hyperparams["latent_dim"])
        self.discriminator = Discriminator(hyperparams["latent_dim"], hyperparams["hidden_dim"])
        self.supervisor = Supervisor(hyperparams["latent_dim"], hyperparams["hidden_dim"], hyperparams["latent_dim"])
        self.embedder = Embedder(self.num_of_features, hyperparams["hidden_dim"], hyperparams["latent_dim"])
        self.recovery = Recovery(hyperparams["latent_dim"], hyperparams["hidden_dim"], self.num_of_features)

    def generate_data(self, num_samples: int):
        noise = generate_noise(num_samples, self.seq_length, self.num_of_features)
        with torch.no_grad():
            generated_embeddings = self.generator(noise)
            supervised_embeddings = self.supervisor(generated_embeddings)
            generated_data = self.recovery(supervised_embeddings)
            return generated_data    
        
    def train(self, data: torch.Tensor):
        self.data_loader = DataLoader(data, hyperparams["batch_size"], shuffle=True)

        # Initialising loss functions
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCELoss()

        # Initialising optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=hyperparams["learning_rate"])
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=hyperparams["learning_rate"])
        self.embedder_optimizer = torch.optim.Adam(self.embedder.parameters(), lr=hyperparams["learning_rate"])
        self.recovery_optimizer = torch.optim.Adam(self.recovery.parameters(), lr=hyperparams["learning_rate"])
        self.supervisor_optimizer = torch.optim.Adam(self.supervisor.parameters(), lr=hyperparams["learning_rate"])

        # Track losses
        self.losses_reconstruction = torch.zeros(hyperparams["epochs"])
        self.losses_supervised = torch.zeros(hyperparams["epochs"])
        self.losses_generator = torch.zeros(hyperparams["epochs"])
        self.losses_embedder = torch.zeros(hyperparams["epochs"])
        self.losses_discriminator = torch.zeros(hyperparams["epochs"])

        # Train
        self.phase_1()
        self.phase_2()
        self.phase_3()

    def phase_1(self):
         # Training phase 1: Minimize reconstruction loss
        print('Start Training Phase 1: Minimize reconstruction loss')
    
        for epoch in range(hyperparams["epochs"]):
            for _, real_sequences in enumerate(self.data_loader):

                # Minimize reconstruction loss
                self.embedder_optimizer.zero_grad()
                self.recovery_optimizer.zero_grad()

                embeddings = self.embedder(real_sequences)
                reconstructions = self.recovery(embeddings)

                reconstruction_loss = self.mse_loss(reconstructions, real_sequences)
                reconstruction_loss.backward()

                self.embedder_optimizer.step()
                self.recovery_optimizer.step()
            
            self.losses_reconstruction[epoch] = reconstruction_loss.item()
            print(f"Epoch {epoch+1}/{hyperparams['epochs']}, Reconstruction Loss: {reconstruction_loss.item()}")
        print("Training phase 1 complete!")
      
    def phase_2(self):
        # Training phase 2: Minimize supervised loss
        print('Start Training Phase 2: Minimize unsupervised loss')

        for epoch in range(hyperparams["epochs"]):
            for _, real_sequences in enumerate(self.data_loader):

                # Train supervisor
                self.supervisor_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()

                embeddings = self.embedder(real_sequences)
                supervised_embeddings = self.supervisor(embeddings)

                supervised_loss = self.mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])
                supervised_loss.backward()

                self.generator_optimizer.step()
                self.supervisor_optimizer.step()
            
            self.losses_supervised[epoch] = supervised_loss.item()
            print(f"Epoch {epoch+1}/{hyperparams['epochs']}, Supervised Loss: {supervised_loss.item()}")
    
    def phase_3(self):
        # Training phase 3: Joint training
        print('Start Training Phase 3: Joint training')

        for epoch in range(hyperparams["epochs"]):
            # Train generator and supervisor twice as much as discriminator
            for _ in range(2):
                for _, real_sequences in enumerate(self.data_loader):
                    # Reset the gradients
                    self.generator_optimizer.zero_grad()
                    self.supervisor_optimizer.zero_grad()
                    self.discriminator_optimizer.zero_grad()
                    self.embedder_optimizer.zero_grad()

                    # Calculate unsupervised loss: noise --> generator --> supervisor --> discriminator --> loss
                    noise = generate_noise(hyperparams["batch_size"], self.seq_length, self.num_of_features)
                    generated_embeddings = self.generator(noise)
                    supervised_embeddings = self.supervisor(generated_embeddings)
                    predictions_fake = self.discriminator(supervised_embeddings)
                    unsupervised_loss = self.bce_loss(predictions_fake, torch.ones_like(predictions_fake))

                    #TODO: Add moments loss

                    # calculate supervised loss: real data --> embedder --> supervisor --> loss
                    embeddings = self.embedder(real_sequences)
                    supervised_embeddings = self.supervisor(embeddings)
                    supervised_loss = self.mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])

                    # Combine losses and calculate gradients
                    generator_loss = supervised_loss + unsupervised_loss
                    generator_loss.backward()

                    # Update generator and supervisor
                    self.generator_optimizer.step()
                    self.supervisor_optimizer.step()

                    # Reset gradients
                    self.embedder_optimizer.zero_grad()
                    self.recovery_optimizer.zero_grad()
                    self.supervisor_optimizer.zero_grad()

                    # Calculate reconstruction loss: real data --> embedder --> recovery --> loss
                    embeddings = self.embedder(real_sequences)
                    reconstructions = self.recovery(embeddings)
                    reconstruction_loss = self.mse_loss(reconstructions, real_sequences)

                    # Calculate supervised_loss: real data --> embedder --> supervisor --> loss
                    supervised_embeddings = self.supervisor(embeddings)
                    supervised_loss = self.mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])

                    # Combine losses and calculate gradients
                    embedder_loss = supervised_loss + reconstruction_loss
                    embedder_loss.backward()

                    # Only update embedder and recovery networks
                    self.embedder_optimizer.step()
                    self.recovery_optimizer.step()

            # Train discriminator
            for _, real_sequences in enumerate(self.data_loader):
                # Reset the gradients of the optimizers
                self.discriminator_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                self.supervisor_optimizer.zero_grad()
                self.embedder_optimizer.zero_grad()

                # Forward pass
                noise = generate_noise(hyperparams["batch_size"], self.seq_length, self.num_of_features)
                fake_embeddings = self.generator(noise)
                supervised_embeddings = self.supervisor(fake_embeddings)
                real_embeddings = self.embedder(real_sequences)

                predictions_fake = self.discriminator(fake_embeddings)
                predictions_real = self.discriminator(real_embeddings)
                predictions_supervised = self.discriminator(supervised_embeddings)

                # Calculate discriminator loss
                loss_fake = self.bce_loss(predictions_fake, torch.zeros_like(predictions_fake))
                loss_supervised = self.bce_loss(predictions_supervised, torch.zeros_like(predictions_supervised))
                loss_real = self.bce_loss(predictions_real, torch.ones_like(predictions_fake))
                discriminator_loss = loss_fake + loss_real + loss_supervised
                discriminator_loss.backward()

                # Only update the discriminator
                self.discriminator_optimizer.step()

            # Track joint losses
            self.losses_discriminator[epoch] = discriminator_loss.item()
            self.losses_generator[epoch] = generator_loss.item()
            self.losses_embedder[epoch] = embedder_loss.item()
            print(f"Epoch {epoch+1}/{hyperparams['epochs']}, Generator loss: {generator_loss.item()}, Embedder loss: {embedder_loss.item()}, Discriminator Loss: {discriminator_loss.item()}")

    def visualise(self, path):
        self.visualise_phase1(path)
        self.visualise_phase2(path)
        self.visualise_phase3(path)

    def visualise_phase1(self, path: str):
        x_axis = np.array(list(range(hyperparams["epochs"])))
        plt.semilogx(x_axis, self.losses_reconstruction.numpy(), label = "Reconstruction loss")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.title(f"TimeGAN Reconstruction Loss")
        plt.legend()
        plt.savefig(f"{path}/reconstruction_loss.png")
        plt.clf()

    def visualise_phase2(self, path: str):
        x_axis = np.array(list(range(hyperparams["epochs"])))
        plt.semilogx(x_axis, self.losses_supervised.numpy(), label = "Supervised loss")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.title(f"TimeGAN Supervised Loss")
        plt.legend()
        plt.savefig(f"{path}/supervised_loss.png")
        plt.clf()

    def visualise_phase3(self, path: str):
        x_axis = np.array(list(range(hyperparams["epochs"])))
        plt.semilogx(x_axis, self.losses_generator.numpy(), label = "Generator Loss")
        plt.semilogx(x_axis, self.losses_discriminator.numpy(), label = "Discriminator Loss")
        plt.semilogx(x_axis, self.losses_embedder.numpy(), label = "Embedder Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.title(f"TimeGAN Joint Losses")
        plt.legend()
        plt.savefig(f"{path}/joint-loss.png")
        plt.clf()

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