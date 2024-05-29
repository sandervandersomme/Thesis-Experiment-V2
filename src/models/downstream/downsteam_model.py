import torch
import torch.nn as nn
from src.utilities.utils import set_device

class DownstreamModel(nn.Module):

    __NAME__ = None

    def __init__(self, **hyperparams):
        super().__init__()

        # torch settings
        self.device = hyperparams.get("device", set_device())

        if "seed" in hyperparams:
            torch.manual_seed(hyperparams["seed"])

        # Training parameters
        self.hidden_dim = hyperparams["hidden_dim"]
        self.num_layers = hyperparams["num_layers"]
        self.batch_size = hyperparams["batch_size"]
        self.epochs = hyperparams["epochs"]
        self.learning_rate = hyperparams['learning_rate']

        # early stoppiung parameters
        self.patience = hyperparams['patience']
        self.min_delta = hyperparams['min_delta']

        # Data dimensions
        self.num_sequences = hyperparams["num_sequences"]
        self.num_events = hyperparams["num_events"]
        self.num_features = hyperparams["num_features"]

        self.output_path = "outputs/"
  
downstream_model_params = {
    "batch_size": 5,
    "hidden_dim": 10,
    "num_layers": 1,
    "epochs": 50,
    "learning_rate": 0.001,
    "patience": 5,
    "min_delta": 0.05,
}