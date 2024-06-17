import torch
import numpy as np

def flatten_into_sequences(data: torch.Tensor):
    """
    Transforms data with shape (sequences, events, features) into date with shape (sequences, features*events)
    Note: Each sequence now is flattened such that different features describe different time-steps"""
    sequences, events, features = data.shape
    return data.reshape(sequences, events*features)

def flatten_into_events(data: torch.Tensor):
    """
    Transforms data with shape (sequences, events, features) into an eventlog with shape (events, features)
    Note: This removes temporal ordering of events
    """
    _, _, features = data.shape
    return data.reshape(-1, features)

def generate_random_data(n_sequence, n_events, n_features):
    return torch.rand((n_sequence, n_events, n_features))
