from src.eval.utility.classification import classification_scores
from src.eval.utility.regression import regression_scores

from src.data.data_loader import create_downstream_data
from src.training.hyperparameters import add_shape_to_params
import torch
from typing import List

def run_downstream_task(task: str, dataset_name: str, syndata: torch.Tensor, train_data: torch.Tensor, test_data: torch.Tensor, columns: List[str], hyperparams: dict, epochs: int, val_split_size: float, seed: int):
    syndata_ds = create_downstream_data(dataset_name, task, syndata, columns)
    train_ds = create_downstream_data(dataset_name, task, train_data, columns)
    test_ds = create_downstream_data(dataset_name, task, test_data, columns)

    hyperparams = add_shape_to_params(hyperparams, train_ds[0][0].shape)

    # Perform task
    if task == "classification":
        return classification_scores(train_ds, syndata_ds, test_ds, epochs, hyperparams, val_split_size, seed)
    elif task == "regression":
        return regression_scores(train_ds, syndata_ds, test_ds, epochs, hyperparams, val_split_size, seed)
        

    raise NotImplementedError
