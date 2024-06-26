from typing import List
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

from src.paths import PATH_CF_165, PATH_CF_184

class CF(Dataset):
    PATH_SEQUENCES = PATH_CF_184
    NAME = "cf"

    def __init__(self):
        super().__init__()
        self.sequences, self.boolean_indices, self.columns = self.load_and_process_data()

    def load_and_process_data(self):
        sequences = pd.read_csv(self.PATH_SEQUENCES)
        columns = [col for col in sequences.columns if col not in ["PATIENTNR", "DATE", "START_KAFTRIO"]]
        sequences, boolean_indices = self.scale_and_reshape(sequences)
        return sequences, boolean_indices, columns
        
    @staticmethod
    def scale_data(sequences):
        scaler = MinMaxScaler()
        return scaler.fit_transform(sequences)

    @staticmethod
    def scale_and_reshape(sequences):
        sequences = sequences.sort_values(by=["PATIENTNR", "DATE"])
        num_events = sequences.groupby("PATIENTNR").size().iloc[0]
        num_sequences = len(sequences) // num_events
        sequences = sequences.drop(columns=["PATIENTNR", "DATE", "START_KAFTRIO"])

        boolean_columns = sequences.select_dtypes(include=['bool']).columns
        boolean_indices = [sequences.columns.get_loc(col) for col in boolean_columns]

        sequences = CF.scale_data(sequences)
        sequences = sequences.reshape(num_sequences, num_events, sequences.shape[1])
        return torch.Tensor(sequences), boolean_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index]

class DownstreamDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor, columns: List[str]):
        self.sequences, self.targets, self.columns = sequences, targets, columns

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]