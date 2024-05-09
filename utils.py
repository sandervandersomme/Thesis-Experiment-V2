import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
from typing import List
import pickle
import argparse

HOSPITAL_COLS = ['DIABETES', 'LIVER_DISEASE', 'PANCREAS', 'PSEUDOMONAS', 
                 'EMERGENCY', 'NEBULIZED', 'ORAL', 'KAFTRIO', 'SYMKEVI', 
                 'ORKAMBI', 'KALYDECO', 'HEIGHT', 'WEIGHT', 'BMI', 'AGE', 
                 'PPFEV1', 'Male', 'F508DEL', 'TIME_SINCE_LAST_EVENT']

def parse_args():
    '''
    Converts arguments into variables
    '''
    print("Parsing arguments..")

    parser = argparse.ArgumentParser()

    # Experiment setup
    parser.add_argument('--num_instances', type=int, help='Number of model instanced that should be generated to account for variance', default=1)
    parser.add_argument('--num_datasets', type=int, help='Number of synthetic datasets generated per instance', default=1)
    parser.add_argument('--num_samples', type=int, help='How many samples should be generated', default=None)

    # Torch settings
    parser.add_argument('--device', type=str, help='Device to be used', default=set_device())
    parser.add_argument('--seed', type=str, help='Seed to be used', default=None)

    args = parser.parse_args()
    return args


def set_device():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device

def generate_noise(batch_size: int, seq_length: int, num_of_features: int):
    """
    Method to generate noise.

    Output: Noise of shape (batch_size, seq_len, input_dim).
    """
    return torch.randn(batch_size, seq_length, num_of_features)

def split_train_test(real_data: torch.Tensor, fake_data: torch.Tensor, learning_rate: float = None, train_size: float = 0.8, hidden_dim: int = 10):
    print("Splitting data into train and test...")

    assert real_data.shape[0] == fake_data.shape[0] # Check if data is balanced (equal number of synthetic and fake sequences)
    assert real_data.shape[1] == fake_data.shape[1] # check if fake and real data have same number of events per sequence
    assert real_data.shape[2] == fake_data.shape[2] # check if fake and real data have same number of features

    labels_real = torch.ones(real_data.shape[0], 1) # create labels for real data
    labels_fake = torch.zeros(fake_data.shape[0], 1) # create labels for fake data

    data = torch.concat([real_data, fake_data], dim=0) # Combine synthetic and real data
    labels = torch.concat([labels_real, labels_fake], dim=0) # Combine labels for synthetic and real data

    test_size = 1 - train_size
    dataset = TensorDataset(data, labels)
    train_data, test_data = random_split(dataset, [train_size, test_size]) # split in train and test data

    return train_data, test_data