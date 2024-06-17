import torch
from torch.utils.data import Dataset, random_split, Subset

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

def split_train_test(dataset: Dataset, train_split: float, generator = None):
    """Returns train and test sequences"""

    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    lengths = [train_size, test_size]
    train_subset, test_subset = random_split(dataset, lengths, generator=generator)

    # train_sequences = dataset.sequences[train_subset.indices]
    # test_sequences = dataset.sequences[test_subset.indices]
    return train_subset, test_subset
