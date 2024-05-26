import torch
from src.utilities.utils import set_device

class GenModel():
    def __init__(self, shape: tuple, **hyperparams):
        """
        - Sets device, seed, batch_size, num_events and num_features
        - Generates noise for generator
        - Generates synthetic data using generator
        """

        # torch settings
        self.device = hyperparams.get("device", set_device())
        
        self.num_sequences, self.num_events, self.num_features = shape

        if "seed" in hyperparams:
            torch.manual_seed(hyperparams["seed"])

        self.epochs = hyperparams["epochs"]
        self.batch_size = hyperparams["batch_size"]
        self.learning_rate = hyperparams["learning_rate"]

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