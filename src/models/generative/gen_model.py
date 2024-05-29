import torch
from src.utilities.utils import set_device

class GenModel():

    __NAME__ = None

    def __init__(self, **hyperparams):
        """
        - Sets device, seed, batch_size, num_events and num_features
        - Generates noise for generator
        - Generates synthetic data using generator
        """

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

        self.output_path = f"outputs/genmodels"
        self.logging_path = "runs/genmodels"

    def generate_noise(self, samples: int):
        """
        Method to generate noise.

        Output: Noise of shape (batch_size, seq_len, input_dim).
        """
        return torch.randn(samples, self.num_events, self.num_features, device=self.device)

    def generate_data(self, num_samples: int) -> torch.Tensor:
        """
        Feeds noise into generator network to generate synthetic data    
        """

        print(f"Generating {num_samples} samples")
        noise = self.generate_noise(num_samples)
        with torch.no_grad():
            return self.generator(noise)
        
gen_params = {
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 10,
    "hidden_dim": 10,
    "num_layers": 1,
    "latent_dim": 10,
    "patience": 5,
    "min_delta": 0.05,
}