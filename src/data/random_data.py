import torch

def generate_random_data(n_sequence, n_events, n_features):
    return torch.rand((n_sequence, n_events, n_features))
