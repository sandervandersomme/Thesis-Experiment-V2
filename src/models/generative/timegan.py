import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.models.generative.gan import Generator, Discriminator
from src.models.generative.gen_model import GenModel

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from src.utilities.early_stopping import EarlyStopping


class Embedder(nn.Module):
    """
    A GRU based embedder that takes in real sequences and converts it into a latent representation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers=1):
        super().__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, sequences):
        output, _ = self.rnn(sequences)
        return self.output_layer(output)
    
class Recovery(nn.Module):
    """
    A GRU based recovery network that takes in a latent representation and returns a reconstructed sample
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers=1):
        super().__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, latent_vector):
        rnn_output, _ = self.rnn(latent_vector)
        return self.output_layer(rnn_output)

class Supervisor(nn.Module):
    """
    A GRU based supervisor that takes in latent representations generated by the generator and returns the sequence generated from the latent representation
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers=1):
        super().__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, latent_representations):
        output, _ = self.rnn(latent_representations)
        return self.output_layer(output)

class TimeGAN(GenModel):
    """
    Time-series GAN consisting of a generator, discriminator, embedder and recovery function.
    """
    
    __NAME__ = "TimeGAN"

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

        # Set hyperparams
        self.latent_dim = hyperparams["latent_dim"]
        self.scaling_factor = hyperparams["scaling_factor"] #10 in paper
        self.gamma_weight = hyperparams["gamma_weight"] # How much does supervised error contribute
        self.disc_loss_threshold = hyperparams["disc_loss_threshold"] # Only update discriminator if it performs badly

        # Create architecture
        self.generator = Generator(self.num_features, self.hidden_dim, self.latent_dim, self.num_layers).to(self.device)
        self.discriminator = Discriminator(self.latent_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.supervisor = Supervisor(self.latent_dim, self.hidden_dim, self.latent_dim, self.num_layers).to(self.device)
        self.embedder = Embedder(self.num_features, self.hidden_dim, self.latent_dim, self.num_layers).to(self.device)
        self.recovery = Recovery(self.latent_dim, self.hidden_dim, self.num_features, self.num_layers).to(self.device)

    def generate_data(self, num_samples: int) -> torch.Tensor:
        noise = self.generate_noise(num_samples)
        with torch.no_grad():
            generated_embeddings = self.generator(noise)
            supervised_embeddings = self.supervisor(generated_embeddings)
            generated_data = self.recovery(supervised_embeddings)
            return generated_data    
        
TimeGAN_params = {
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 10,
    "hidden_dim": 10,
    "num_layers": 1,
    "latent_dim": 10,
    "patience": 5,
    "min_delta": 0.05,
    "scaling_factor": 10,
    "gamma_weight": 1,
    "disc_loss_threshold": 0.15
}

def train_TimeGAN(model: TimeGAN, train_data: torch.Tensor, val_data: Dataset, log_dir):
    # Initialising optimizers
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=model.learning_rate)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=model.learning_rate)
    embedder_optimizer = torch.optim.Adam(model.embedder.parameters(), lr=model.learning_rate)
    recovery_optimizer = torch.optim.Adam(model.recovery.parameters(), lr=model.learning_rate)
    supervisor_optimizer = torch.optim.Adam(model.supervisor.parameters(), lr=model.learning_rate)

    # Setup dataloaders
    val_loader = DataLoader(val_data, batch_size=model.batch_size, shuffle=False)
    train_loader = DataLoader(train_data, model.batch_size, shuffle=True)

    # Initialising loss functions
    mse_loss = torch.nn.MSELoss().to(model.device)
    bce_loss = torch.nn.BCELoss().to(model.device)

    # Setup hyperparameter tuning
    best_val_loss = float('inf')

    # Set model to training mode
    model.generator.train()
    model.supervisor.train()
    model.embedder.train()
    model.recovery.train()
    model.discriminator.train()

    # Train
    train_embedding_network(model, train_loader, val_loader, embedder_optimizer, recovery_optimizer, mse_loss, log_dir=f"{log_dir}/embedding/")
    train_supervised(model, train_loader, val_loader, supervisor_optimizer, generator_optimizer, mse_loss, log_dir=f"{log_dir}/supervised/")
    best_val_loss = train_joint(model, train_loader, supervisor_optimizer, generator_optimizer, discriminator_optimizer, embedder_optimizer, recovery_optimizer, bce_loss, mse_loss, log_dir=f"{log_dir}/joint/")

    return best_val_loss

def train_embedding_network(model: TimeGAN, train_loader: DataLoader, val_loader: DataLoader, embedder_optimizer: torch.optim.Adam, recovery_optimizer: torch.optim.Adam, mse_loss: torch.nn.MSELoss, log_dir:str):
    writer = SummaryWriter(log_dir)
    
    # Train embedder and recovery: minimize construction loss
    print('Start Training Phase 1: Minimize reconstruction loss')

    # Track losses
    train_losses = []
    val_losses = []

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    for epoch in range(model.epochs):
        # Train model
        model.embedder.train()
        model.recovery.train()
        loss = train_autoencoder(model, train_loader, mse_loss, embedder_optimizer, recovery_optimizer)
        train_losses.append(loss.item())
        
        # Validate model
        model.recovery.eval()
        model.embedder.eval()
        val_loss = validate_autoencoder(model, val_loader, mse_loss)  
        val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}/{model.epochs}, reconstruction Loss: {loss.item()} val_loss: {val_loss}")

        # Check if best loss has increased
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            writer.close()
            break

        # Log losses to TensorBoard
        writer.add_scalar("Loss/reconstruction loss", loss, epoch)
        writer.add_scalar("Loss/critic_real", val_loss, epoch)

    print("Training phase 1 complete!")

    writer.close()

    plot_reconstruction_loss(f"{model.output_path}/{model.__NAME__}", train_losses, val_losses)

def train_supervised(model: TimeGAN, train_loader: DataLoader, val_loader: DataLoader, sup_optim: torch.optim.Adam, emb_optim: torch.optim.Adam, mse_loss: torch.nn.MSELoss, log_dir:str):
    writer = SummaryWriter(log_dir)
    
    # Training phase 2: Minimize supervised loss
    print('Start Training Phase 2: Minimize unsupervised loss')

    train_losses = []
    val_losses = []

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    for epoch in range(model.epochs):
        # Train model
        model.embedder.train()
        model.supervisor.train()
        loss = train_supervisor_embedder(model, train_loader, mse_loss, sup_optim, emb_optim)
        train_losses.append(loss.item())

        # Validate model
        model.supervisor.eval()
        model.embedder.eval()
        val_loss = validate_supervisor(model, val_loader, mse_loss)  
        val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}/{model.epochs}, Supervised Loss: {loss.item()} val_loss: {val_loss}")

        # Check if best loss has increased
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            writer.close()
            break

        # Log losses to TensorBoard
        writer.add_scalar("Loss/supervisor loss", loss, epoch)
        writer.add_scalar("Loss/val loss", val_loss, epoch)

    writer.close()

    plot_supervised_loss(f"{model.output_path}/{model.__NAME__}", train_losses, val_losses)
    
def train_joint(model: TimeGAN, train_loader: DataLoader, supervisor_optimizer: torch.optim.Adam, generator_optimizer: torch.optim.Adam, discriminator_optimizer: torch.optim.Adam, embedder_optimizer: torch.optim.Adam, recovery_optimizer: torch.optim.Adam, bce_loss: torch.nn.BCELoss, mse_loss: torch.nn.BCELoss, log_dir:str):
    writer = SummaryWriter(log_dir)
    
    # Training phase 3: Joint training
    print('Start Training Phase 3: Joint training')

    gen_losses = []
    emb_losses = []
    disc_losses = []
    val_losses = []

    # setup early stopping
    early_stopping = EarlyStopping(model.patience, model.min_delta)
    best_val_loss = float('inf')

    epoch_idx2 = 0
    for epoch in range(model.epochs):

        # Train generator and supervisor twice as much as discriminator
        for _ in range(2):

            # Train generator
            model.generator.train()
            model.supervisor.train()

            gen_loss = train_generator(model, train_loader, bce_loss, mse_loss, generator_optimizer, supervisor_optimizer)
            gen_losses.append(gen_loss.item())
             
            # Train embedder
            model.embedder.train()
            model.recovery.train()

            emb_loss = train_embedder(model, train_loader, mse_loss, embedder_optimizer, recovery_optimizer)
            emb_losses.append(emb_loss.item())

            # Validate model
            model.generator.eval()
            model.embedder.eval()
            model.discriminator.eval()
            val_loss = validate_generator(model, bce_loss)
            val_losses.append(val_loss.item())

            # Log losses to TensorBoard
            writer.add_scalar("Loss/generator loss", gen_loss, epoch_idx2)
            writer.add_scalar("Loss/embedder loss", emb_loss, epoch_idx2)
            writer.add_scalar("Loss/validation loss", val_loss, epoch_idx2) 

            epoch_idx2 += 1

        # Train discriminator
        model.discriminator.train()
        model.generator.train()
        model.embedder.train()
        disc_loss = train_discriminator(model, train_loader, bce_loss, discriminator_optimizer)
        disc_losses.append(disc_loss.item())

        # Check if best loss has increased
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            writer.close()
            break
        
        writer.add_scalar("Loss/discriminator loss", disc_loss, epoch)

        print(f"Epoch {epoch+1}/{model.epochs}, Generator loss: {gen_loss.item()}, Embedder loss: {emb_loss.item()}, Discriminator Loss: {disc_loss.item()}, val loss: {val_loss}")

    writer.close()

    plot_disc_losses(f"{model.output_path}/{model.__NAME__}", disc_losses, val_losses)
    plot_joint_losses(f"{model.output_path}/{model.__NAME__}", gen_losses, emb_losses, val_losses)

    return best_val_loss

def train_autoencoder(model: TimeGAN, dataloader: DataLoader, mse_loss: torch.nn.MSELoss, emb_optim: torch.optim.Adam, rec_optim: torch.optim.Adam):
    total_loss = 0.0

    for _, sequences in enumerate(dataloader):
        sequences = sequences.to(model.device)

        # Forward pass
        embeddings = model.embedder(sequences)
        reconstructions = model.recovery(embeddings)
        loss = model.scaling_factor * torch.sqrt(mse_loss(reconstructions, sequences))
    
        # Backpropagate
        emb_optim.zero_grad()
        rec_optim.zero_grad()
        loss.backward(retain_graph=True)
        emb_optim.step()
        rec_optim.step()

        total_loss += loss

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_supervisor_embedder(model: TimeGAN, dataloader: DataLoader, mse_loss: torch.nn.MSELoss, sup_optim, emb_optim):
    total_loss = 0.0
    for _, sequences in enumerate(dataloader):
        sequences = sequences.to(model.device)

        # Forward pass
        embeddings = model.embedder(sequences)
        supervised_embeddings = model.supervisor(embeddings)
        supervised_loss = mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])

        # Backpropagate
        sup_optim.zero_grad()
        emb_optim.zero_grad()
        supervised_loss.backward(retain_graph=True)
        emb_optim.step()
        sup_optim.step()

        total_loss += supervised_loss
    avg_loss = total_loss / len(dataloader)

    return avg_loss

def train_generator(model: TimeGAN, dataloader: DataLoader, bce_loss: torch.nn.BCELoss, mse_loss: torch.nn.MSELoss, gen_optim: torch.optim.Adam, supervisor_optim: torch.optim.Adam):
    total_loss = 0.0
    for _, sequences in enumerate(dataloader):
        sequences = sequences.to(model.device)

        # Forward pass Generator
        gen_loss = generator_loss(model, sequences, bce_loss, mse_loss)
        total_loss += gen_loss

        # Backpropagate
        gen_optim.zero_grad()
        supervisor_optim.zero_grad()
        gen_loss.backward(retain_graph=True)
        gen_optim.step()
        supervisor_optim.step()

        total_loss += gen_loss
    
    avg_loss = total_loss / (len(dataloader))

    return avg_loss

def train_embedder(model: TimeGAN, dataloader: DataLoader, mse_loss: torch.nn.MSELoss, emb_optim: torch.optim.Adam, rec_optim: torch.optim.Adam):
    total_loss = 0.0
    for _, sequences in enumerate(dataloader):  
        sequences = sequences.to(model.device)

        # Forward pass embedder
        emb_loss = embedder_loss(model, sequences, mse_loss)
        total_loss += emb_loss

        # Backpropagate
        emb_optim.zero_grad()
        rec_optim.zero_grad()
        emb_loss.backward(retain_graph=True)
        emb_optim.step()
        rec_optim.step()
        
        avg_loss = total_loss / (len(dataloader))
    return avg_loss

def train_discriminator(model: TimeGAN, dataloader: DataLoader, bce_loss: torch.nn.BCELoss, disc_optim: torch.optim.Adam):
    total_loss = 0.0
    for _, sequences in enumerate(dataloader):
        sequences = sequences.to(model.device)

        # Forward pass
        disc_loss = discriminator_loss(model, sequences, bce_loss)
        total_loss += disc_loss

        disc_optim.zero_grad()
        if disc_loss > model.disc_loss_threshold:
        # Backpropagate
            disc_loss.backward(retain_graph=True)
            disc_optim.step()
    avg_loss = total_loss / len(dataloader)

    return avg_loss

def generator_loss(model: TimeGAN, sequences: torch.Tensor, bce_loss: torch.nn.BCELoss, mse_loss: torch.nn.MSELoss):
    noise = model.generate_noise(model.batch_size).to(model.device)
    E_hat = model.generator(noise)
    H_hat = model.supervisor(E_hat)
    H = model.embedder(sequences) 
    H_hat_supervised = model.supervisor(H)
    X_hat = model.recovery(H_hat)

    # Adversarial loss: noise (Z) --> generator (E_hat) --> supervisor (H_hat) --> discriminator (Y_fake) --> adversarial loss (G_loss_U)
    Y_fake = model.discriminator(H_hat)
    G_loss_U = bce_loss(Y_fake, torch.ones_like(Y_fake).to(model.device))

    # Adversarial loss: noise (Z) --> generator (E_hat) --> discriminator (Y_fake_e) --> Adversarial loss (G_loss_U_e)
    Y_fake_e = model.discriminator(E_hat)
    G_loss_U_e = bce_loss(Y_fake_e, torch.ones_like(Y_fake_e).to(model.device))

    # Supervised loss: real data (X) --> embedder (H) --> supervisor (H_hat_supervise) --> supervised_loss (G_loss_S)
    G_Loss_S = mse_loss(H_hat_supervised[:,:-1,:], H[:,1:,:])

    # Moments Losses:
    G_loss_v1 = torch.mean(torch.abs(torch.std(X_hat , dim=[0,1], unbiased=False) - torch.std(sequences , dim=[0,1], unbiased=False))) # Maybe add 1e-6 to avoid division by 0
    G_loss_V2 = torch.mean(torch.abs(torch.mean(X_hat, dim=[0,1]) - torch.mean(sequences, dim=[0,1])))

    # Combine losses and calculate gradients
    generator_loss = G_loss_U + G_loss_U_e + G_Loss_S + G_loss_v1 + G_loss_V2 # TODO: Add constants as hyperparameters

    return generator_loss

def embedder_loss(model: TimeGAN, sequences: torch.Tensor, mse_loss: torch.nn.MSELoss):
    # Calculate reconstruction loss: real data (X) --> embedder (X) --> Supervisor (H_hat_supervise) --> loss
    H = model.embedder(sequences)
    H_hat_supervise = model.supervisor(H)
    X_tilde = model.recovery(H)

    # Losses
    E_loss_0 = 10 * torch.sqrt(mse_loss(X_tilde, sequences))
    G_loss_S = mse_loss(H_hat_supervise[:, :-1, :], H[:,1:,:])
    E_loss = E_loss_0 + 0.1 * G_loss_S

    return E_loss

def validate_autoencoder(model: TimeGAN, val_loader: DataLoader, mse_loss: torch.nn.MSELoss):
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for _, sequences in enumerate(val_loader):
            sequences = sequences.to(model.device)

            embeddings = model.embedder(sequences)
            reconstructions = model.recovery(embeddings)
            loss = model.scaling_factor * torch.sqrt(mse_loss(reconstructions, sequences))

            total_loss += loss

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss

def validate_supervisor(model: TimeGAN, val_loader: DataLoader, mse_loss: torch.nn.MSELoss):
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for _, sequences in enumerate(val_loader):
            sequences = sequences.to(model.device)

            embeddings = model.embedder(sequences)
            supervised_embeddings = model.supervisor(embeddings)
            supervised_loss = mse_loss(embeddings[:,1:,:], supervised_embeddings[:,:-1,:])

            total_loss += supervised_loss
    avg_loss = total_loss / len(val_loader)

    return avg_loss

def validate_generator(model: TimeGAN, bce_loss: torch.nn.BCELoss):
    with torch.no_grad():
        # Forward pass Generator
        noise = model.generate_noise(model.batch_size).to(model.device)
        E_hat = model.generator(noise)
        H_hat = model.supervisor(E_hat)

        # Adversarial loss: noise (Z) --> generator (E_hat) --> supervisor (H_hat) --> discriminator (Y_fake) --> adversarial loss (G_loss_U)
        Y_fake = model.discriminator(H_hat)
        G_loss_U = bce_loss(Y_fake, torch.ones_like(Y_fake).to(model.device))

        # Adversarial loss: noise (Z) --> generator (E_hat) --> discriminator (Y_fake_e) --> Adversarial loss (G_loss_U_e)
        Y_fake_e = model.discriminator(E_hat)
        G_loss_U_e = bce_loss(Y_fake_e, torch.ones_like(Y_fake_e).to(model.device))

        # Combine losses and calculate gradients
        generator_loss = G_loss_U + G_loss_U_e

    return generator_loss

def discriminator_loss(model: TimeGAN, sequences: torch.Tensor, bce_loss: torch.nn.BCELoss):
    noise = model.generate_noise(len(sequences))
    E_hat = model.generator(noise)
    H_Hat = model.supervisor(E_hat)
    H = model.embedder(sequences)
    x_hat = model.recovery(H_Hat)

    Y_real = model.discriminator(H)
    Y_fake_e = model.discriminator(E_hat)
    Y_fake = model.discriminator(H_Hat)

    # Calculate discriminator loss
    loss_real = bce_loss(Y_real, torch.ones_like(Y_real).to(model.device))
    loss_fake = bce_loss(Y_fake, torch.zeros_like(Y_fake).to(model.device))
    loss_fake_e = bce_loss(Y_fake_e, torch.zeros_like(Y_fake_e))

    discriminator_loss = loss_fake + loss_real + loss_fake_e * model.gamma_weight
    return discriminator_loss

def plot_reconstruction_loss(path: str, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o', linestyle='-', label = "Reconstruction Loss")
    plt.plot(val_losses, marker='o', linestyle='-', label = "Validation Loss")
    # plt.plot(train_losses.numpy(), marker='o', linestyle='-', label = "Reconstruction Loss")
    # plt.plot(train_losses.numpy(), marker='o', linestyle='-', label = "Reconstruction Loss")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{path}/reconstruction")  # Save the figure to a file
    plt.close()

def plot_supervised_loss(path: str, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o', linestyle='-', label = "Supervised Loss")
    plt.plot(val_losses, marker='o', linestyle='-', label = "Validation Loss")
    # plt.plot(supervised_loss.numpy(), marker='o', linestyle='-', label = "Supervised Loss")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{path}/supervised")  # Save the figure to a file
    plt.close()

def plot_disc_losses(path: str, losses_disc, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_disc, marker='o', linestyle='-', label = "Discriminator Loss")

    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{path}/disc")  # Save the figure to a file
    plt.close()


def plot_joint_losses(path: str, losses_gen, losses_emb, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_gen, marker='o', linestyle='-', label = "Generator Loss")
    plt.plot(losses_emb, marker='o', linestyle='-', label = "Embedder Loss")
    plt.plot(val_losses, marker='o', linestyle='-', label = "Validation Loss")


    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(f"{path}/Joint")  # Save the figure to a file
    plt.close()

