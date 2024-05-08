import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from GAN import Generator, Discriminator
from utils import generate_noise
import matplotlib.pyplot as plt
from parameters import hyperparams

class RGAN():
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    __MODEL__ = "RGAN"
        
    def __init__(self, shape: tuple, device: str, seed: int = None):
        self.device = device

        # set seed for reproducibility
        if seed:
            torch.manual_seed(seed)

        # set dimensions
        self.num_of_sequences = shape[0]
        self.seq_length = shape[1]
        self.num_of_features = shape[2]

        # Create architecture
        self.generator = Generator(self.num_of_features, hyperparams['hidden_dim'], self.num_of_features)
        self.discriminator = Discriminator(self.num_of_features, hyperparams['hidden_dim'])

    def train(self, data: torch.Tensor):
        data_loader = DataLoader(data, 
                                 batch_size=int(hyperparams['batch_size']/2), 
                                 shuffle=True)

        # Set optimizers
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), 
                                                    lr=hyperparams['learning_rate'])
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), 
                                                        lr=hyperparams['learning_rate'])
        
        # Set loss function
        self.criterion = nn.BCELoss().to(self.device)

        # Track losses per epochs
        self.losses_generator = torch.zeros(hyperparams['epochs'])
        self.losses_discriminator = torch.zeros(hyperparams['epochs'])

        for epoch_id in range(hyperparams['epochs']):
            for _, real_sequences in enumerate(data_loader):

                self.losses_discriminator[epoch_id] = self.train_discriminator(real_sequences)
                self.losses_generator[epoch_id] = self.train_generator()

            print(f"Epoch {epoch_id+1}/{hyperparams['epochs']}, Loss D.: {self.losses_discriminator[epoch_id].item()}, Loss G.: {self.losses_generator[epoch_id].item()}")
        
    def train_generator(self):
        # Reset the gradients of the optimizers
        self.optimizer_generator.zero_grad()
        self.discriminator.zero_grad()

        # Forward Generator Pass
        noise = generate_noise(hyperparams["batch_size"], self.seq_length, self.num_of_features).to(self.device)
        fake_data = self.generator(noise)
        predictions_fake = self.discriminator(fake_data)

        # Calculate generator loss
        labels_real = torch.ones_like(predictions_fake)
        loss_generator = self.criterion(predictions_fake, labels_real)
        loss_generator.backward()
        self.optimizer_generator.step()

        # Track loss
        return loss_generator.item()

    def train_discriminator(self, real_sequences):
        # reset gradients of generator and discriminator
        self.generator.zero_grad()
        self.discriminator.zero_grad()

        # Forward pass
        noise = generate_noise(int(hyperparams["batch_size"]/2), self.seq_length, self.num_of_features).to(self.device)
        fake_data = self.generator(noise)
        predictions_real = self.discriminator(real_sequences)
        predictions_fake = self.discriminator(fake_data)

        # Calculate discriminator loss
        labels_fake = torch.zeros_like(predictions_fake)
        labels_real = torch.ones_like(predictions_real)
        loss_real = self.criterion(predictions_real, labels_real)
        loss_fake = self.criterion(predictions_fake, labels_fake)
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        self.optimizer_discriminator.step()

        # Track loss
        return loss_discriminator.item()
  
    def visualise(self, path):
        x_axis = np.array(list(range(self.epochs)))
        plt.semilogx(x_axis, self.losses_generator.numpy(), label = "Generator Loss")
        plt.semilogx(x_axis, self.losses_discriminator.numpy(), label = "Discriminator Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.title(f"RGAN Losses")
        plt.legend()
        plt.savefig(f"{path}/loss.png")
        plt.clf()
    
    def generate_data(self, num_samples: int, path: str):
        print(f"Generating {num_samples} samples")
        noise = generate_noise(num_samples, self.seq_length, self.num_of_features)
        with torch.no_grad():
            generated_data = self.generator(noise)
            np.save(f"{path}/{num_samples}", generated_data.numpy())
            return generated_data
