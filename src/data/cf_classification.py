from src.data.cf import DownstreamDataset
import torch
from typing import List

def create_cf_classification_data(sequences: torch.Tensor, columns: List[str], name: str):
    target_var = columns.index("PPFEV1")
    labels = (sequences[:, -1, target_var] > sequences[:, 0, target_var]).long().unsqueeze(dim=1)

    sequences = torch.cat((sequences[:, :, :target_var], sequences[:, :, target_var+1:]), dim=2)
    columns = [col for col in columns if col != "PPFEV1"]
    
    return DownstreamDataset(sequences, labels, columns, name)

if __name__ == "__main__":
    from src.data.cf import CF
    from src.data.data_processing import split_train_test
    from src.data.data_loader import save_to_csv

    # Load real data and split
    cf = CF()
    train_sequences, test_sequences = split_train_test(cf, 0.7)
    
    # Create downstream datasets
    cf_class_train = create_cf_classification_data(train_sequences, cf.columns, "Classification_Real_Train")
    cf_class_test = create_cf_classification_data(test_sequences, cf.columns, "Classification_Real_Test")

    save_to_csv(cf_class_train.sequences, cf_class_train.targets, cf_class_train.columns, "outputs/test")
    
