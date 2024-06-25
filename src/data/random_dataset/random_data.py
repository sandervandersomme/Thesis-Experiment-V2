import torch
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, num_sequences, num_events, num_features):
        self.sequences = torch.rand(num_sequences, num_events, num_features)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index]