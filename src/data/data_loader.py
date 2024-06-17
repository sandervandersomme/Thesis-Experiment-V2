from typing import List
import torch
import numpy as np
import pandas as pd

from src.data.cf import CF
from torch.utils.data import Dataset

def load_syn_data(path): return torch.load(path)

def select_data(dataset: str) -> Dataset:
    "select and load dataset"
    if dataset == "cf":
        return CF()

# def create_downstream_dataset(dataset, task: str):
#     if task == "classification":
#         return create_classification_dataset(dataset)
#     if task == "regression":
#         return create_regression_dataset(dataset)
    
# def create_classification_dataset(dataset):
#     if dataset.NAME == "cf":
#         return CF_Classification(dataset.sequences, dataset.columns.copy())

# def create_regression_dataset(dataset):
#     if dataset.NAME == "cf":
#         return CF_Regression(dataset.sequences, dataset.columns.copy())
    
def save_to_csv(sequences: torch.Tensor, labels: torch.Tensor, columns: List[str], filename: str):
    sequences_np = sequences.numpy()
    labels_np = labels.numpy()

    num_sequences, num_events, num_features = sequences_np.shape
    sequences_reshaped = sequences_np.reshape(num_sequences * num_events, num_features)

    df_sequences = pd.DataFrame(sequences_reshaped, columns=columns)
    sequence_ids = np.repeat(np.arange(num_sequences), num_events)
    df_sequences['SEQUENCE_ID'] = sequence_ids

    df_labels = pd.DataFrame(np.repeat(labels_np, num_events), columns=['LABEL'])
    df_combined = pd.concat([df_sequences, df_labels], axis=1)

    df_combined.to_csv(filename, index=False)
    print(f"Data saved to {filename}")