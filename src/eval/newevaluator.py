import torch
from torch.utils.data import Dataset
from typing import List
from src.models.gen_model import GenModel

class Evaluator():
    def __init__(self, eval_dir, eval_args) -> None:
        self.eval_dir = eval_dir
        self.eval_args = eval_args

        # Collect results
        self.results = None
        self.average_results = None
        self.num_datasets = 0

    def evaluate_dataset(self, dataset: torch.Tensor, dataset_id: int, model: GenModel, model_id: int, real_data, train_indices, test_indices):
        raise NotImplementedError

