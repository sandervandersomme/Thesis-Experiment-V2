from src.eval.newevaluator import Evaluator
from src.models.gen_model import GenModel


import torch


class FidelityEvaluator(Evaluator):
    def __init__(self, eval_args, eval_dir: str) -> None:
        super().__init__(eval_args, eval_dir)

    def evaluate_dataset(self, dataset: torch.Tensor, dataset_id: int, model: GenModel, model_id: int):
        print("It's working")
