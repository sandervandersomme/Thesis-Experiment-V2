from src.data.cf.cf import DownstreamDataset
import torch
from typing import List

def create_sepsis_classification_data(sequences: torch.Tensor, columns: List[str]):
    label_idx = columns.index("Return ER")
    labels = sequences[:, -1, label_idx].float().unsqueeze(dim=1)

    sequences = torch.cat((sequences[:, :, :label_idx], sequences[:, :, label_idx+1:]), dim=2)
    columns = [col for col in columns if col != "Return ER"]
    
    return DownstreamDataset(sequences, labels, columns)

if __name__ == "__main__":
    from src.data.sepsis.sepsis import Sepsis

    sepsis = Sepsis()
    sepsis_class_train = create_sepsis_classification_data(sepsis.sequences, sepsis.columns)
