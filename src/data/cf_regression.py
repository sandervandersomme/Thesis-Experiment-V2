import torch
from torch.utils.data import Dataset

from typing import List

class CF_Regression(Dataset):

    NAME = "cf_regression"

    def __init__(self, sequences: torch.Tensor, columns: List):
        super().__init__()

        target_var = columns.index("PPFEV1")
        self.labels = sequences[:,-1,target_var].unsqueeze(dim=1)

        # Remove ppfev1 from sequences
        self.sequences = torch.cat((sequences[:,:,:target_var], 
                                    sequences[:,:, target_var+1:]), dim=2)
        self.columns = columns.remove("PPFEV1")
        
        assert len(self.sequences) == len(self.labels)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]
    


