import torch
import torch.nn as nn


from src.models.gan import Generator, Discriminator
from src.models.gen_model import GenModel
from src.training.early_stopping import EarlyStopping

class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers=1):
        super().__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())


    def forward(self, noise: torch.Tensor):
        "Input: noise of shape (batch_size, num_of_features)"
        rnn_out, _ = self.rnn(noise)
        return self.output_layer(rnn_out)



class TGAN(GenModel):
    """
    A tabular GAN consisting of a generator and discriminator.
    """

    NAME = "rgan"
        
    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)

        # Create architecture
        self.generator = Generator(self.num_features, self.hidden_dim, self.num_features, self.num_layers).to(self.device)
        self.discriminator = Discriminator(self.num_features, self.hidden_dim, self.num_layers).to(self.device)



