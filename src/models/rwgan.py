import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.gen_model import GenModel
from src.models.gan import Generator

from typing import Callable

class RWGAN(GenModel):
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    __MODEL__ = "RWGAN"
        
    def __init__(self, shape: tuple, **hyperparams):
        super().__init__(shape, **hyperparams)
        
        self.clip_value = hyperparams["clip_value"]
        self.n_critic = hyperparams["n_critic"]

        # Create architecture
        self.generator = Generator(self.num_features, self.hidden_dim, self.num_features).to(self.device)
        self.discriminator = Critic(self.num_features, self.hidden_dim).to(self.device)
        

class ClipConstraint():
    """
    Clips tensor values
    """
    def __init__(self, clip_value: float) -> None:
        self.clip_value = clip_value

    def __call__(self, model) -> torch.Any:
        for p in model.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

RWGAN_params = {
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 10,
    "hidden_dim": 10,
    "num_layers": 1,
    "n_critic": 1,
    "clip_value": 0.05
}

def train_RWGAN(model: RWGAN, data: torch.Tensor, path: str):
    data_loader = DataLoader(data, batch_size=model.batch_size, shuffle=True)

    # set the loss function, optimizers and clipping constraint
    optimizer_generator = torch.optim.RMSprop(model.generator.parameters(), lr=model.learning_rate)
    optimizer_critic = torch.optim.RMSprop(model.discriminator.parameters(), lr=model.learning_rate)
    criterion = wasserstein_loss
    clip = ClipConstraint(model.clip_value)

    # Track losses per epochs
    losses_generator = torch.zeros(model.epochs)
    losses_critic_real = torch.zeros(model.epochs)
    losses_critic_fake = torch.zeros(model.epochs)

    for epoch in range(model.epochs):
        for _, sequences in enumerate(data_loader):       
            sequences = sequences.to(model.device)      

            losses_critic_real[epoch], losses_critic_fake[epoch] = train_critic(model, sequences, optimizer_critic, criterion, clip)
            losses_generator[epoch] = train_generator(model, optimizer_generator, criterion)

        print(f"Epoch {epoch+1}/{model.epochs}, Loss C-real: {losses_critic_real[epoch].item()}, Loss D-fake: {losses_critic_fake[epoch].item()}, Loss G.: {losses_generator[epoch].item()}")

    visualise(path, losses_generator, losses_critic_real, losses_critic_fake)

def train_generator(model: RWGAN, optimizer_generator: torch.optim.RMSprop, criterion: Callable):
    # Reset the gradients of the optimizers
    optimizer_generator.zero_grad()
    model.discriminator.zero_grad()

    # Forward pass
    noise = model.generate_noise(model.batch_size).to(model.device)
    fake_data = model.generator(noise)
    predictions_fake = model.discriminator(fake_data)

    # Calculate loss for generator
    labels_real = -torch.ones_like(predictions_fake).to(model.device)
    loss_generator = criterion(predictions_fake, labels_real)
    loss_generator.backward()
    optimizer_generator.step()

    # Track loss
    return loss_generator.item()

def train_critic(model: RWGAN, sequences, optimizer_critic: torch.optim.RMSprop, criterion: Callable, clip: ClipConstraint):
    for iteration_id in (range(model.n_critic)):
        # Track losses
        temp_losses_real = torch.zeros(model.n_critic).to(model.device)
        temp_losses_fake = torch.zeros(model.n_critic).to(model.device)

        # reset gradients of generator and discriminator
        model.generator.zero_grad()
        model.discriminator.zero_grad()

        # Forward Generator pass
        noise = model.generate_noise(model.batch_size).to(model.device)
        fake_data = model.generator(noise)

        # Forward discriminator pass
        predictions_real = model.discriminator(sequences)
        predictions_fake = model.discriminator(fake_data)
        labels_fake = torch.ones_like(predictions_fake).to(model.device)
        labels_real = -torch.ones_like(predictions_real).to(model.device)

        # Calculate discriminator loss
        loss_real = criterion(predictions_real, labels_real)
        loss_fake = criterion(predictions_fake, labels_fake)
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_critic.step()

        # Clip gradients
        clip(model.discriminator)

        # Track loss
        temp_losses_real[iteration_id] = loss_real.item()
        temp_losses_fake[iteration_id] = loss_fake.item()

    critic_loss_real = temp_losses_real.mean()
    critic_loss_fake = temp_losses_fake.mean()

    return critic_loss_real, critic_loss_fake

def visualise(path, losses_generator, loss_critic_real, loss_critic_fake):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_generator, marker='o', linestyle='-', label="generator loss")
    plt.plot(loss_critic_real, marker='o', linestyle='-', label="critic_loss_real")
    plt.plot(loss_critic_fake, marker='o', linestyle='-', label="critic_loss_fake")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)  # Save the figure to a file
    plt.clf()


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


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)