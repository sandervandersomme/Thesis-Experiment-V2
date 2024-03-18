import os
import pandas as pd
import numpy as np
import argparse
import torch

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
    parser.add_argument('--dataset', type=str, help='What dataset should be used?', default='dummy', choices=['cf', 'sepsis', 'dummy'])
    parser.add_argument('--model', type=str, help='What model should be used?', default='RGAN', choices=['RGAN'])
    parser.add_argument('--device', type=str, help='Device to be used', default=set_device())
    parser.add_argument('--seed', type=str, help='Seed to be used', default=None)
    parser.add_argument('--epochs', type=int, help='Number of epochs for training', default=100)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--input_dim', type=int, help='Dimensions of input layer of generator', default=10)
    parser.add_argument('--hidden_dim', type=int, help='Dimensions of hidden layer', default=20)
    parser.add_argument('--latent_dim', type=int, help='Dimensions of the latent representation of TimeGAN', default=10)
    parser.add_argument('--batch_size', type=int, help='Number of sequences to be processed each iteration', default=1)


    args = parser.parse_args()
    settings = vars(parser.parse_args())
    return args, settings

def set_device():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device

def load_data(dataset: str):
    '''
    Loads data from numpy file
    '''
    print("Loading " + dataset + " data..")

    path = "data/" + dataset + ".npy"

    if dataset == "dummy":
        if not os.path.exists(path):
            create_dummy_data(path)
        
    data = np.load(path)
    return torch.from_numpy(data).float()

def create_dummy_data(path):
    '''
    Generates dummy dataset
    '''
    
    print("Generating dummy data..")
    arr = np.random.randint(0, 101, size=(20,10, 4))
    np.save(path, arr)

def generate_noise(batch_size, seq_length, num_of_features):
    """
    Method to generate noise.

    Output: Noise of shape (batch_size, seq_len, input_dim).
    """
    return torch.randn(batch_size, seq_length, num_of_features)