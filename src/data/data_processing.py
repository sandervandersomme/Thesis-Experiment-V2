import torch
from torch.utils.data import Dataset, random_split

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

def split_train_val_test(data: Dataset, train_split: float, val_split: float):
    train_size = int(len(data) * train_split)
    val_size = int(len(data) * val_split)
    test_size = len(data) - train_size - val_size
    lengths = [train_size, val_size, test_size]
    return random_split(data, lengths)
