from typing import List
from torch.utils.data import Dataset
import torch
import pandas as pd

class Sepsis(Dataset):
    NAME = "sepsis"

    def __init__(self):
        super().__init__()
        self.sequences = None
        self.columns = None

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index]

