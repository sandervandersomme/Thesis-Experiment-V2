import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from parameters import hyperparams

from models.GAN import Generator
from utils import generate_noise

class RWGAN():
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    __MODEL__ = "RWGAN"
        
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
        self.generator = Generator(self.num_of_features, hyperparams["hidden_dim"], self.num_of_features)
        self.discriminator = Critic(self.num_of_features, hyperparams["hidden_dim"])

    def train(self, data: torch.Tensor):
        data_loader = DataLoader(data, batch_size=hyperparams["batch_size"], shuffle=True)

        # set the loss function, optimizers and clipping constraint
        self.optimizer_generator = torch.optim.RMSprop(self.generator.parameters(), lr=hyperparams["learning_rate"])
        self.optimizer_discriminator = torch.optim.RMSprop(self.discriminator.parameters(), lr=hyperparams["learning_rate"])
        self.criterion = wasserstein_loss
        self.clip = ClipConstraint(hyperparams["clip_value"])

        # Track losses per epochs
        self.losses_generator = torch.zeros(hyperparams["epochs"])
        self.losses_critic_real = torch.zeros(hyperparams["epochs"])
        self.losses_critic_fake = torch.zeros(hyperparams["epochs"])

        for epoch_id in range(hyperparams["epochs"]):
            for _, real_sequences in enumerate(data_loader):             
                self.losses_critic_real[epoch_id], self.losses_critic_real[epoch_id] = self.train_critic(real_sequences, hyperparams["crtic_iterations"])
                self.losses_generator[epoch_id] = self.train_generator()

            print(f"Epoch {epoch_id+1}/{hyperparams['epochs']}, Loss C-real: {self.losses_critic_real[epoch_id].item()}, Loss D-fake: {self.losses_critic_fake[epoch_id].item()}, Loss G.: {self.losses_generator[epoch_id].item()}")
    
    def train_generator(self):
        # Reset the gradients of the optimizers
        self.optimizer_generator.zero_grad()
        self.discriminator.zero_grad()

        # Forward pass
        noise = generate_noise(hyperparams["batch_size"], self.seq_length, self.num_of_features).to(self.device)
        fake_data = self.generator(noise)
        predictions_fake = self.discriminator(fake_data)

        # Calculate loss for generator
        labels_real = -torch.ones_like(predictions_fake)
        loss_generator = self.criterion(predictions_fake, labels_real)
        loss_generator.backward()
        self.optimizer_generator.step()

        # Track loss
        return loss_generator.item()

    def train_critic(self, real_sequences, n_critic):
        for iteration_id in (range(hyperparams["critic_iterations"])):
            # Track losses
            temp_losses_real = torch.zeros(n_critic)
            temp_losses_fake = torch.zeros(n_critic)

            # reset gradients of generator and discriminator
            self.generator.zero_grad()
            self.discriminator.zero_grad()

            # Forward Generator pass
            noise = generate_noise(hyperparams["batch_size"], self.seq_length, self.num_of_features).to(self.device)
            fake_data = self.generator(noise)

            # Forward discriminator pass
            predictions_real = self.discriminator(real_sequences)
            predictions_fake = self.discriminator(fake_data)
            labels_fake = torch.ones_like(predictions_fake)
            labels_real = -torch.ones_like(predictions_real)

            # Calculate discriminator loss
            loss_real = self.criterion(predictions_real, labels_real)
            loss_fake = self.criterion(predictions_fake, labels_fake)
            loss_discriminator = loss_real + loss_fake
            loss_discriminator.backward()
            self.optimizer_discriminator.step()

            # Clip gradients
            self.clip(self.discriminator)

            # Track loss
            temp_losses_real[iteration_id] = loss_real.item()
            temp_losses_fake[iteration_id] = loss_fake.item()

        critic_loss_real = temp_losses_real.mean()
        critic_loss_fake = temp_losses_fake.mean()

        return critic_loss_real, critic_loss_fake

    def visualise(self, path):
        x_axis = np.array(list(range(hyperparams["epochs"])))
        plt.semilogx(x_axis, self.losses_generator.numpy(), label = "Generator Loss")
        plt.semilogx(x_axis, self.losses_critic_real.numpy(), label = "Critic Loss real")
        plt.semilogx(x_axis, self.losses_critic_fake.numpy(), label = "Critic Loss fake")

        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.title(f"RWGAN Losses")
        plt.legend()
        plt.savefig(f"{path}/loss.png")
        plt.clf()

    def generate_data(self, num_samples: int):
        print(f"Generating {num_samples} samples")
        noise = generate_noise(num_samples, self.seq_length, self.num_of_features)
        with torch.no_grad():
            return self.generator(noise)

class Critic(nn.Module):
    """
    A GRU based critic that takes in a sequence as input and returns the likelihood of it being synthetic or real.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(Critic, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)


    def forward(self, sequences: torch.Tensor):
        rnn_output, _ = self.rnn(sequences)
        return self.output_layer(rnn_output)

class ClipConstraint():
    """
    Clips tensor values
    """
    def __init__(self) -> None:
        self.clip_value = hyperparams["clip_value"]

    def __call__(self, model) -> torch.Any:
        for p in model.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)