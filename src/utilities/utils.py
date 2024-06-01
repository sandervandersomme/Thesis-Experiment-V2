import argparse
import torch
import numpy as np
import json
import os

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
    print(f"Using {device} device")
    return device

def handle_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    if isinstance(obj, np.generic):
        return obj.item()

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