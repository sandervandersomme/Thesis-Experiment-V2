import torch
from torch.utils.data import Dataset
from typing import List

class CF_Classification(Dataset):

    NAME = "cf_classification"

    def __init__(self, sequences: torch.Tensor, columns = List[str]):
        super().__init__()
        self.sequences = sequences
        self.columns = columns

        target_var = self.columns.index("PPFEV1")
        self.labels = set_labels(self.sequences, target_var).unsqueeze(dim=1)
        
        assert len(self.sequences) == len(self.labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]
    
    
def set_labels(sequences: torch.Tensor, target_var):
    first_events = sequences[:, 0, target_var]
    last_events = sequences[:, -1, target_var]
    labels = last_events > first_events

    return labels.float()
