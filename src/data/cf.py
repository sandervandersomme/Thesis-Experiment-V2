from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
import pandas as pd

class CF(Dataset):
    PATH_SEQUENCES = "datasets/processed/cf_full.csv"
    NAME = "cf"

    def __init__(self):
        super().__init__()

        # Load data
        self.sequences = pd.read_csv(self.PATH_SEQUENCES)
        self.columns = [col for col in self.sequences.columns if col not in ["PATIENTNR", "DATE", "START_KAFTRIO"]]

        #  data
        self.sequences = scale_and_reshape(self.sequences)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index]
    
def scale_data(sequences):
    scaler = MinMaxScaler()
    sequences = scaler.fit_transform(sequences)
    return sequences

def scale_and_reshape(sequences):
    sequences = sequences.sort_values(by=["PATIENTNR", "DATE"])

    # Get dimensions
    num_events = sequences.groupby("PATIENTNR").size().iloc[0]
    num_sequences = int(len(sequences) / num_events)

    # Remove redundant columns
    sequences.drop(columns=["PATIENTNR", "DATE", "START_KAFTRIO"], inplace=True)

    # scale data
    sequences = scale_data(sequences)
    sequences = sequences.reshape(num_sequences, num_events, sequences.shape[1])

    # Update sequences
    
    # Convert data into tensors
    sequences = torch.Tensor(sequences)
    return sequences