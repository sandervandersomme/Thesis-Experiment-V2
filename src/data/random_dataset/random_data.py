import torch
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, num_sequences=20, num_events=4, num_features=20):
        self.sequences = torch.rand(num_sequences, num_events, num_features)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index]
    
class RandomDownstreamRegressionDataset(Dataset):
    def __init__(self, num_sequences=20, num_events=4, num_features=20):
        self.sequences = torch.rand(num_sequences, num_events, num_features)
        self.targets = torch.rand((num_sequences, 1))
        self.columns = [f"column {id}" for id in range(num_features)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]
    
class RandomDownstreamClassificationDataset(Dataset):
    def __init__(self, num_sequences=20, num_events=4, num_features=20):
        self.sequences = torch.rand(num_sequences, num_events, num_features)
        self.targets = torch.randint(0, 2, size=(num_sequences, 1), dtype=torch.float32)
        self.columns = [f"column {id}" for id in range(num_features)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]