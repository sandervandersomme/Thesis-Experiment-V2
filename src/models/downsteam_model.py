import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from src.utilities.utils import set_device

class DownstreamModel(nn.Module):

    __NAME__ = None

    def __init__(self, shape: tuple, **hyperparams):
        super().__init__()

        # torch settings
        if "device" in hyperparams: self.device = hyperparams["device"]
        else: self.device = set_device()

        if "seed" in hyperparams:
            torch.manual_seed(hyperparams["seed"])

        self.num_sequences, self.num_events, self.num_features = shape

        # Training parameters
        self.hidden_dim = hyperparams["hidden_dim"]
        self.num_layers = hyperparams["num_layers"]
        self.batch_size = hyperparams["batch_size"]
        self.epochs = hyperparams["epochs"]
        self.learning_rate = hyperparams['learning_rate']
    
    # Implement something like early stopping
  