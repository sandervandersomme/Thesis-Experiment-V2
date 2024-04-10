import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from GAN import Generator, Discriminator
from utils import generate_noise, visualise

class RGAN():
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    __MODEL__ = "RGAN"
        
    def __init__(self, data: torch.Tensor, device: str, hidden_dim: int, seed: int = None, batch_size = 1):
        self.device = device
        self.data = data
        self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # set seed for reproducibility
        if seed:
            torch.manual_seed(seed)

        # set dimensions
        self.num_of_sequences = self.data.size(0)
        self.seq_length = self.data.size(1)
        self.num_of_features = self.data.size(2)
        self.batch_size = batch_size

        # Create architecture
        self.generator = Generator(self.num_of_features, hidden_dim, self.num_of_features)
        self.discriminator = Discriminator(self.num_of_features, hidden_dim)

    def train(self, epochs: int, learning_rate: float):
        # set the loss function and optimizers
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        criterion = nn.BCELoss().to(self.device)

        # Track losses per epochs
        losses_generator = torch.zeros(epochs)
        losses_discriminator = torch.zeros(epochs)

        for epoch in range(epochs):
            for _, real_sequences in enumerate(self.data_loader):
                # reset gradients of generator and discriminator
                self.generator.zero_grad()
                self.discriminator.zero_grad()

                # Forward pass
                noise = generate_noise(self.batch_size, self.seq_length, self.num_of_features).to(self.device)
                fake_data = self.generator(noise)
                predictions_real = self.discriminator(real_sequences)
                predictions_fake = self.discriminator(fake_data)

                # Calculate discriminator loss
                labels_fake = torch.zeros_like(predictions_fake)
                labels_real = torch.ones_like(predictions_real)
                loss_real = criterion(predictions_real, labels_real)
                loss_fake = criterion(predictions_fake, labels_fake)
                loss_discriminator = loss_real + loss_fake
                loss_discriminator.backward()

                # Update discriminator
                optimizer_discriminator.step()

                # Reset the gradients of the optimizers
                optimizer_generator.zero_grad()
                self.discriminator.zero_grad()

                # Forward pass
                noise = generate_noise(self.batch_size, self.seq_length, self.num_of_features).to(self.device)
                fake_data = self.generator(noise)
                predictions_fake = self.discriminator(fake_data)

                # Calculate loss for generator
                labels_real = torch.ones_like(predictions_fake)
                loss_generator = criterion(predictions_fake, labels_real)
                loss_generator.backward()

                # Update generator
                optimizer_generator.step()

            losses_discriminator[epoch] = loss_discriminator.item()
            losses_generator[epoch] = loss_generator.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss D.: {loss_discriminator.item()}, Loss G.: {loss_generator.item()}")
        
        visualise(epochs, [losses_discriminator, losses_generator], "Epochs", "Loss", ["Loss dicriminator", "Loss generator"], "RGAN losses")
    
    def generate_data(self, num_samples: int, data_path: str, epochs: int):
        noise = generate_noise(num_samples, self.seq_length, self.num_of_features)
        with torch.no_grad():
            generated_data = self.generator(noise)
            np.save(f"output/syndata/{self.__MODEL__}/{data_path}_{epochs}", generated_data.numpy())
            return generated_data


if __name__ == '__main__':
    from utils import set_device

    # load testing data
    path = "dummy.npy"
    data = torch.from_numpy(np.load("data/" + path)).float()
    print(data.shape)

    # Setting parameters
    device = set_device()

    print("Creating RGAN..")
    model = RGAN(data, device, hidden_dim=10)

    print("Training RGAN..")
    model.train(epochs=10, batch_size=1, learning_rate=0.001)
    gen_data = model.generate_data(10, path)

    print("Finished!")
