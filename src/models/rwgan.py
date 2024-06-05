import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.models.gen_model import GenModel
from src.models.gan import Generator
from src.training.early_stopping import EarlyStopping

class Critic(nn.Module):
    """
    A GRU based critic that takes in a sequence as input and returns the likelihood of it being synthetic or real.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers=1):
        super().__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: torch.Tensor):
        rnn_output, _ = self.rnn(sequences)
        return self.output_layer(rnn_output[:, -1, :])

class RWGAN(GenModel):
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    NAME = "rwgan"
        
    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)
        
        self.clip_value = hyperparams["clip_value"]
        self.n_critic = hyperparams["n_critic"]

        # Create architecture
        self.generator = Generator(self.num_features, self.hidden_dim, self.num_features, self.num_layers).to(self.device)
        self.critic = Critic(self.num_features, self.hidden_dim, self.num_layers).to(self.device)
        
class ClipConstraint():
    """
    Clips tensor values
    """
    def __init__(self, clip_value: float) -> None:
        self.clip_value = clip_value

    def __call__(self, model) -> torch.Any:
        for p in model.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

def train_RWGAN(model: RWGAN, train_data: torch.Tensor, log_run_dir: str, log_loss_dir: str):
    writer = SummaryWriter(log_run_dir)

    # Setup training
    optimizer_generator = torch.optim.RMSprop(model.generator.parameters(), lr=model.learning_rate)
    optimizer_critic = torch.optim.RMSprop(model.critic.parameters(), lr=model.learning_rate)
    train_loader = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)
    clip = ClipConstraint(model.clip_value)

    # setup early stopping and hyperparameter tuning
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    # Loss tracking
    gen_losses = []
    critic_losses = []
    val_losses = []

    # Start training loop
    for epoch in range(model.epochs):
        model.generator.train()
        model.critic.train()

        # Calculate losses
        gen_loss, critic_loss = train_loss(train_loader, model, optimizer_generator, optimizer_critic, clip) 
        gen_losses.append(gen_loss)
        # critic_losses_real.append(critic_loss_real.item())
        # critic_losses_fake.append(critic_loss_fake.item())
        critic_losses.append(critic_loss.item())

        # Validate model with generator loss
        model.generator.eval()
        model.critic.eval()
        val_loss = validation_loss(model)
        val_losses.append(val_loss.item())

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            writer.close()
            break

        # Check if best loss has increased
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Log losses to TensorBoard
        writer.add_scalar("Loss/gen", gen_loss, epoch)
        writer.add_scalar("Loss/critic_real", critic_loss, epoch)
        # writer.add_scalar("Loss/critic_fake", critic_loss_fake, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Epoch {epoch+1}/{model.epochs}, Loss C-real: {critic_loss}, Loss G.: {gen_loss}, val loss: {val_loss}")

    writer.close()

    plot_losses(f"{log_loss_dir}loss.png", gen_losses, critic_losses)

    return best_val_loss

def train_loss(train_loader: DataLoader, model: RWGAN, gen_optim: torch.optim.RMSprop, critic_optim: torch.optim.RMSprop, clip: ClipConstraint):
    critic_loss = 0.0
    loss_generator = 0.0
    
    for _, sequences in enumerate(train_loader):       
        sequences = sequences.to(model.device)      

        critic_loss = train_critic(model, sequences, critic_optim, clip) # Returns (real loss, fake loss)
        loss_generator += train_generator(model, gen_optim)

    avg_gen_loss = loss_generator / len(train_loader)
    avg_crit_loss = critic_loss / len(train_loader)

    return avg_gen_loss, avg_crit_loss

def validation_loss(model: RWGAN):
    """Function for calculating the validation loss based on the generator's performance in fooling the discriminator"""
    # generate fake data
    noise = model.generate_noise(model.batch_size).to(model.device)
    fake_data = model.generator(noise)
    predictions_fake = model.critic(fake_data)

    # Calculate loss for generator
    loss = -predictions_fake.mean()
    return loss # TODO Maybe add .item() if error is returned

def train_generator(model: RWGAN, optimizer_generator: torch.optim.RMSprop):
    # Reset the gradients of the optimizers
    optimizer_generator.zero_grad()

    # Forward pass
    noise = model.generate_noise(model.batch_size).to(model.device)
    fake_data = model.generator(noise)
    predictions_fake = model.critic(fake_data)

    # Calculate loss for generator
    loss = predictions_fake.mean()
    loss.backward()
    optimizer_generator.step()

    # Track loss
    return loss.item()

def train_critic(model: RWGAN, sequences, optimizer_critic: torch.optim.RMSprop, clip: ClipConstraint):
    for iteration_id in (range(model.n_critic)):
        # Track losses
        temp_losses = torch.zeros(model.n_critic).to(model.device)

        # reset gradients of generator and discriminator
        model.critic.zero_grad()

        # Forward Generator pass
        noise = model.generate_noise(model.batch_size).to(model.device)
        fake_data = model.generator(noise)

        # Forward discriminator pass
        predictions_real = model.critic(sequences)
        predictions_fake = model.critic(fake_data)

        loss = (predictions_real.mean() - predictions_fake.mean())
        # loss = wasserstein_loss() + wasserstein_loss()

        # Calculate discriminator loss
        loss.backward(retain_graph=True)
        optimizer_critic.step()

        # Clip gradients
        clip(model.critic)

        # Track loss
        temp_losses[iteration_id] = loss.item()

    critic_loss = temp_losses.mean()

    return critic_loss

def plot_losses(path, losses_generator, losses_critic_real):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_generator, marker='o', linestyle='-', label="generator loss")
    plt.plot(losses_critic_real, marker='o', linestyle='-', label="critic_loss_real")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)  # Save the figure to a file
    plt.close()