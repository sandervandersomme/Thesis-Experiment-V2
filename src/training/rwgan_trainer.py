import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.models import GenModel
from src.training.trainer import Trainer
from src.utilities.utils import set_device

class RWGANTrainer(Trainer):
    def __init__(self, model: GenModel, train_data: torch.Tensor, validate=False, logging_path=None, loss_path=None, device=set_device(), stop_early=True, verbose=False):
        super().__init__(model, train_data, validate, logging_path, loss_path, device, stop_early, verbose)

        # Setup training
        self.optimizer_generator = torch.optim.RMSprop(model.generator.parameters(), lr=model.learning_rate)
        self.optimizer_critic = torch.optim.RMSprop(model.critic.parameters(), lr=model.learning_rate)
        self.train_loader = DataLoader(train_data, batch_size=model.batch_size, shuffle=True)

        # Loss tracking
        self.gen_losses = []
        self.critic_losses = []

    def train(self, epochs, verbose=False):
        # Start training loop
        for epoch in range(epochs):
            
            self.model.generator.train()
            self.model.critic.train()

            self.train_networks()

            # Model validation
            if self.validate or self.early_stopping:
                val_loss = self.validation_loss()

            # Check for early stopping
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    if self.writer:
                        self.writer.close()
                    break

            if self.verbose:
                print(f"Epoch {self.epoch+1}/{epochs}, Loss D.: {self.disc_losses[self.epoch]}, Loss G.: {self.gen_losses[self.epoch]}")
            self.epoch += 1

        if self.writer:
            self.writer.close()

        if self.loss_path:
            plot_losses(f"{self.loss_path}loss.png", self.gen_losses, self.disc_losses)

    def train_networks(self):
        critic_loss = 0.0
        loss_generator = 0.0
        
        for _, sequences in enumerate(self.train_loader):       
            sequences = sequences.to(model.device)      

            critic_loss = self.train_critic(sequences)
            loss_generator += self.train_generator()

        avg_gen_loss = loss_generator / len(self.train_loader)
        avg_crit_loss = critic_loss / len(self.train_loader)

        self.gen_losses.append(avg_gen_loss)
        self.critic_losses.append(avg_crit_loss)

        if self.writer:
            self.writer.add_scalar('Loss/disc', avg_crit_loss, self.epoch)
            self.writer.add_scalar('Loss/gen', avg_gen_loss, self.epoch)
    
    def train_generator(self):
        # Reset the gradients of the optimizers
        self.optimizer_generator.zero_grad()

        # Forward pass
        noise = model.generate_noise(model.batch_size).to(model.device)
        fake_data = model.generator(noise)
        predictions_fake = model.critic(fake_data)

        # Calculate loss for generator
        loss = predictions_fake.mean()
        loss.backward()
        self.optimizer_generator.step()

        # Track loss
        return loss.item()
    
    def train_critic(self, sequences):
        for iteration_id in (range(model.n_critic)):
            temp_losses = torch.zeros(model.n_critic).to(model.device)
            model.critic.zero_grad()

            # Forward Generator pass
            noise = model.generate_noise(model.batch_size).to(model.device)
            fake_data = model.generator(noise)

            # Forward discriminator pass
            predictions_real = model.critic(sequences)
            predictions_fake = model.critic(fake_data)
            loss = (predictions_real.mean() - predictions_fake.mean())
            loss.backward(retain_graph=True)
            self.optimizer_critic.step()

            # Clip gradients
            self.model.clip(model.critic)

            # Track loss
            temp_losses[iteration_id] = loss.item()

        critic_loss = temp_losses.mean()

        return critic_loss.item()
    
    def validation_loss(self):
        """Function for calculating the validation loss based on the generator's performance in fooling the discriminator"""
        # generate fake data
        noise = self.model.generate_noise(self.model.batch_size).to(self.model.device)
        fake_data = self.model.generator(noise)
        predictions_fake = self.model.critic(fake_data)

        # Calculate loss for generator
        loss = -predictions_fake.mean()
        return loss # TODO Maybe add .item() if error is returned   
  
def plot_losses(path, train_loss_gen, train_loss_disc):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_gen, marker='o', linestyle='-', label='train loss generator')
    plt.plot(train_loss_disc, marker='o', linestyle='-', label='train loss discriminator')

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)  # Save the figure to a file
    plt.close()
    
if __name__ == "__main__":
    from src.data.data_loader import select_data
    dataset = select_data('cf')

    from src.models.models import RWGAN
    from src.training.hyperparameters import select_hyperparams

    hyperparams = select_hyperparams('cf', 'rgan', dataset.sequences.shape)
    model = RWGAN(**hyperparams)
    trainer = RWGANTrainer(model, dataset, verbose=True)
    trainer.train(50)