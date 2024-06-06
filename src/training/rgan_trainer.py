import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.models.models import GenModel
from src.training.trainer import Trainer
from src.utilities.utils import set_device

class RGANTrainer(Trainer):
    def __init__(self, model: GenModel, train_data: torch.Tensor, validate=False, logging_path=None, loss_path=None, device=set_device(), stop_early=True, verbose=False):
        super().__init__(model, train_data, validate, logging_path, loss_path, device, stop_early, verbose)

        # Setup training
        self.half_batch = int(self.model.batch_size/2)
        self.train_loader = DataLoader(self.train_data, batch_size=self.half_batch, shuffle=True)
        self.optimizer_generator = torch.optim.Adam(self.model.generator.parameters(), lr=self.model.learning_rate)
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.model.learning_rate)
        self.criterion = nn.BCELoss().to(self.model.device)

        # Loss tracking
        self.gen_losses = []
        self.disc_losses = []
        self.val_losses = []

    def train(self, epochs):
        # Start training loop
        for epoch in range(epochs):
            
            self.model.generator.train()
            self.model.discriminator.train()

            self.train_networks()

            # Model validation
            if self.validate or self.stop_early:
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
        # Initialize losses
        disc_loss = 0.0
        gen_loss = 0.0

        for _, sequences in enumerate(self.train_loader):
            sequences = sequences.to(self.model.device)

            disc_loss += self.train_discriminator(sequences)
            gen_loss += self.train_generator()

        avg_disc_loss = disc_loss / len(self.train_loader)
        avg_gen_loss = gen_loss / len(self.train_loader)

        self.gen_losses.append(avg_gen_loss)
        self.disc_losses.append(avg_disc_loss)

        if self.writer:
            self.writer.add_scalar('Loss/disc', avg_disc_loss, self.epoch)
            self.writer.add_scalar('Loss/gen', avg_gen_loss, self.epoch)
    
    def train_generator(self):
        # Forward Generator Pass
        noise = self.model.generate_noise(self.model.batch_size).to(self.device)
        fake_data = self.model.generator(noise)
        predictions_fake = self.model.discriminator(fake_data)

        # Calculate generator loss
        labels_real = torch.ones_like(predictions_fake).to(self.device)
        loss_generator = self.criterion(predictions_fake, labels_real)

        # Backpropagation
        self.optimizer_generator.zero_grad()
        loss_generator.backward()
        self.optimizer_generator.step()

        return loss_generator.item()
    
    def train_discriminator(self, sequences):
        # Forward pass
        noise = self.model.generate_noise(int(self.model.batch_size/2)).to(self.model.device)
        fake_data = self.model.generator(noise)
        predictions_real = self.model.discriminator(sequences)
        predictions_fake = self.model.discriminator(fake_data)

        # Calculate discriminator loss
        labels_fake = torch.zeros_like(predictions_fake).to(self.model.device)
        labels_real = torch.ones_like(predictions_real).to(self.model.device)

        loss_real = self.criterion(predictions_real, labels_real)
        loss_fake = self.criterion(predictions_fake, labels_fake)
        loss_discriminator = loss_real + loss_fake
        
        # Backpropogation
        self.optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        self.optimizer_discriminator.step()

        # Track loss
        return loss_discriminator.item()
    
    def validation_loss(self):
        self.model.generator.eval()
        self.model.discriminator.eval()

        # Forward Generator Pass
        noise = self.model.generate_noise(self.model.batch_size).to(self.device)
        fake_data = self.model.generator(noise)
        predictions_fake = self.model.discriminator(fake_data)

        # Calculate generator loss
        labels_real = torch.ones_like(predictions_fake).to(self.device)
        val_loss = self.criterion(predictions_fake, labels_real).item()
        self.val_losses.append(val_loss)

        super().validation(val_loss)
        return val_loss
  
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

    from src.models.models import RGAN
    from src.training.hyperparameters import select_hyperparams

    hyperparams = select_hyperparams('cf', 'rgan', dataset.sequences.shape)
    model = RGAN(**hyperparams)
    trainer = RGANTrainer(model, dataset, verbose=True, stop_early=False)
    trainer.train(50)