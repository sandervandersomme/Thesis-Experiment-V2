import torch
import torch.nn as nn
import numpy as np
from GAN import Generator, Discriminator
from utils import generate_noise

class RGAN():
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    __MODEL__ = "RGAN"
        
    def __init__(self, data: torch.Tensor, device: str, input_dim: int, hidden_dim: int, batch_size: int, seed: int):
        self.device = device
        self.data = data

        # set seed for reproducibility
        if seed:
            torch.manual_seed(seed)

        # set dimensions
        self.num_of_sequences, self.seq_length, self.num_of_features = self.data.shape

        # Create architecture
        self.generator = Generator(input_dim, hidden_dim, self.num_of_features)
        self.discriminator = Discriminator(self.num_of_features, hidden_dim)

    def train(self, epochs: int, learning_rate, batch_size):
        # set the loss function for the discriminator
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        criterion = nn.BCELoss().to(self.device)

        for epoch in range(epochs):
            for i in range(0, self.num_of_sequences, batch_size):
                
                # reset gradients of generator and discriminator
                self.generator.zero_grad()
                self.discriminator.zero_grad()

                # Create databatch and labels
                real_data = self.data[i:i+batch_size].to(self.device)
                fake_labels = torch.zeros(real_data.shape[0], real_data.shape[1], 1).to(self.device)
                real_labels = torch.ones(real_data.size(0), real_data.shape[1], 1).to(self.device)

                # Create fake data for discriminator
                noise = generate_noise().to(self.device)
                fake_data = self.generator(noise)

                # Train the discriminator
                outputs_real = self.discriminator(real_data)
                outputs_fake = self.discriminator(fake_data)
                
                # Calculate discriminator loss
                loss_real = criterion(outputs_real, real_labels)
                loss_fake = criterion(outputs_fake, fake_labels)
                loss_discriminator = loss_real + loss_fake

                # Update discriminator
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                # Create fake data for training generator
                noise = generate_noise().to(self.device)
                fake_data = self.generator(noise)

                # Calculate loss for generator
                outputs_fake = self.discriminator(fake_data)
                loss_generator = criterion(outputs_fake, real_labels)
                loss_generator.backward()

                # Update generator
                self.optimizer_generator.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss D.: {loss_discriminator.item()}, Loss G.: {loss_generator.item()}")
    
    def generate_data(self, num_samples, data_path):
        noise = generate_noise(num_samples, self.seq_length, self.num_of_features)
        with torch.no_grad():
            generated_data = self.generator(noise)
            np.save("data/output/"+ data_path + "_" + self.__MODEL__, generated_data.numpy())
            return generated_data

if __name__ == '__main__':
    from utils import set_device

    # load testing data
    path = "data/dummy.npy"
    data = torch.from_numpy(np.load(path)).float()
    print(data.shape)

    # Setting parameters
    device = set_device()
    input_dim = 10
    hidden_dim = data.shape[2]

    print("Creating RGAN..")
    model = RGAN(data, device, input_dim, hidden_dim)

    print("Training RGAN..")
    model.train(epochs=10, learning_rate=0.001)
    gen_data = model.generate_data(10)
    print(gen_data)

    print("Finished!")
