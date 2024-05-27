import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from src.models.generative.gan import Generator, Discriminator
from src.models.generative.gen_model import GenModel
from src.utilities.early_stopping import EarlyStopping

class RGAN(GenModel):
    """
    A Recurrent GAN consisting of a generator and discriminator.
    """

    __NAME__ = "RGAN"
        
    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

        # Create architecture
        self.generator = Generator(self.num_features, self.hidden_dim, self.num_features, self.num_layers).to(self.device)
        self.discriminator = Discriminator(self.num_features, self.hidden_dim, self.num_layers).to(self.device)

RGAN_params = {
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 200,
    "hidden_dim": 10,
    "num_layers": 1
}

def train_RGAN(model: RGAN, train_data: torch.Tensor, val_data: Dataset, log_dir):
    writer = SummaryWriter(log_dir)

    # Setup training
    half_batch = int(model.batch_size/2)
    train_loader = DataLoader(train_data, batch_size=half_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=model.batch_size, shuffle=False)
    optimizer_generator = torch.optim.Adam(model.generator.parameters(), lr=model.learning_rate)
    optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=model.learning_rate)

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    # Set loss function
    criterion = nn.BCELoss().to(model.device)

    # Loss tracking
    gen_train_losses = []
    gen_val_losses = []
    disc_train_losses = []

    for epoch in range(model.epochs):

        # Model training
        model.generator.train()
        model.discriminator.train()

        disc_loss = 0.0
        gen_loss = 0.0
        for _, sequences in enumerate(train_loader):
            sequences = sequences.to(model.device)

            disc_loss += discriminator_loss(model, optimizer_discriminator, criterion, sequences)
            gen_loss += generator_loss(model, optimizer_generator, criterion)

        avg_disc_loss = disc_loss / len(train_loader)
        avg_gen_loss = gen_loss / len(train_loader)

        disc_train_losses.append(avg_disc_loss)
        gen_train_losses.append(avg_gen_loss)

        # Model validation
        model.generator.eval()
        model.discriminator.eval()
        g_val_loss = validate(model, criterion)
        gen_val_losses.append(g_val_loss)

        # Check if best loss has increased
        if g_val_loss < best_val_loss:
            best_val_loss = g_val_loss

        # Check for early stopping
        early_stopping(g_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            writer.close()
            break

        # Logging losses
        writer.add_scalar('Loss/Discriminator_Train', avg_disc_loss, epoch)
        writer.add_scalar('Loss/Generator_Train', avg_gen_loss, epoch)
        writer.add_scalar('Loss/Generator_Val', g_val_loss, epoch)

        print(f"Epoch {epoch+1}/{model.epochs}, Loss D.: {avg_disc_loss[epoch].item()}, Loss G.: {avg_gen_loss[epoch].item()}")
    
    writer.close()

    plot_losses(f"{model.output_path}/rgan-loss", gen_train_losses, disc_train_losses, gen_val_losses)

    return best_val_loss
    
def validate(model: RGAN, loss_fn:  nn.BCELoss):
    # generate fake data
    noise = model.generate_noise(model.batch_size).to(model.device)
    fake_data = model.generator(noise)

    # Run fake data through discriminator
    predictions_fake = model.discriminator(fake_data)

    # Set labels for generator data (inversed, because the loss should indicate how many the discriminator doesn't classify as real)
    labels_real = torch.ones_like(predictions_fake).to(model.device)
    return loss_fn(predictions_fake, labels_real)

def generator_loss(model: RGAN, optimizer_generator: torch.optim.Adam, criterion: nn.BCELoss):
    # Forward Generator Pass
    noise = model.generate_noise(model.batch_size).to(model.device)
    fake_data = model.generator(noise)
    predictions_fake = model.discriminator(fake_data)

    # Calculate generator loss
    labels_real = torch.ones_like(predictions_fake).to(model.device)
    loss_generator = criterion(predictions_fake, labels_real)

    # Backpropagation
    optimizer_generator.zero_grad()
    loss_generator.backward()
    optimizer_generator.step()

    # Track loss
    return loss_generator.item()

def discriminator_loss(model: GenModel, optimizer_discriminator: torch.optim.Adam, criterion: nn.BCELoss, sequences):
    # Forward pass
    noise = model.generate_noise(int(model.batch_size/2)).to(model.device)
    fake_data = model.generator(noise)
    predictions_real = model.discriminator(sequences)
    predictions_fake = model.discriminator(fake_data)

    # Calculate discriminator loss
    labels_fake = torch.zeros_like(predictions_fake).to(model.device)
    labels_real = torch.ones_like(predictions_real).to(model.device)

    loss_real = criterion(predictions_real, labels_real)
    loss_fake = criterion(predictions_fake, labels_fake)
    loss_discriminator = loss_real + loss_fake
    
    # Backpropogation
    optimizer_discriminator.zero_grad()
    loss_discriminator.backward()
    optimizer_discriminator.step()

    # Track loss
    return loss_discriminator.item()
  
def plot_losses(path, train_loss_gen, train_loss_disc, val_loss_gen):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_gen, marker='o', linestyle='-', label='train loss generator')
    plt.plot(train_loss_disc, marker='o', linestyle='-', label='train loss discriminator')
    plt.plot(val_loss_gen, marker='o', linestyle='-', label='val loss generator')

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(path)  # Save the figure to a file
    plt.close()
    


