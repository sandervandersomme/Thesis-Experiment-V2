import torch
from typing import List
from src.models.gen_model import GenModel

class Evaluator():
    def __init__(self, eval_dir, eval_args) -> None:
        self.eval_dir = eval_dir
        self.eval_args = eval_args

        # Collect results
        self.results = None
        self.average_results = None

    def evaluate_dataset(self, dataset: torch.Tensor, dataset_id: int, model: GenModel, model_id: int):
        pass

