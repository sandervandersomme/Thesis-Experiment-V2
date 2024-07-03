from typing import List
from torch.utils.data import Dataset
import torch
import pandas as pd

class Sepsis(Dataset):
    path_data = "datasets/processed/sepsis/sepsis.csv"
    NAME = "sepsis"

    def __init__(self):
        super().__init__()
        self.sequences = pd.read_csv(self.path_data)
        self.columns = self.sequences.columns
        self.boolean_indices = self.extract_boolean_indices()
        self.sequences = self.reshape()

    def reshape(self):
        num_events = self.sequences.groupby("CaseID").size().iloc[0]
        num_sequences = len(self.sequences) // num_events
        self.sequences.drop(columns=["CaseID"], inplace=True)

        self.sequences = self.sequences.values.reshape(num_sequences, num_events, self.sequences.shape[1]).astype(float)
        return torch.Tensor(self.sequences)

    def extract_boolean_indices(self):
        boolean_columns = self.sequences.select_dtypes(include=['bool']).columns
        boolean_indices = [self.sequences.columns.get_loc(col) for col in boolean_columns]
        return boolean_indices


    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index]

if __name__ =="__main__":
    print(Sepsis().sequences.shape)