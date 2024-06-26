import argparse
import torch
import numpy as np
import json
import os
import pickle
import pandas as pd

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
    parser.add_argument('--device', type=str, help='Device to be used')
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
    return device

def handle_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    if isinstance(obj, np.generic):
        return obj.item()

def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_best_params_and_score(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            best_trial = json.load(f)
        return best_trial
    else:
        return None

def save_trial(trial, path):
    best_trial_data = {
        'params': trial.params,
        'value': trial.value
    }
    with open(path, 'w') as f:
        json.dump(best_trial_data, f)

def calculate_split_lengths(dataset, train_size: 0.8):
    train_samples = int(len(dataset) * train_size)
    return [train_samples, len(dataset)-train_samples]

def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def load_syndata(path):
    if os.path.exists(path):
        return torch.load(path)

def save_df_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(f'{path}.csv', index=False)

def save_matrix_to_np(matrix: np.ndarray, path: str):
    np.save(path, matrix)