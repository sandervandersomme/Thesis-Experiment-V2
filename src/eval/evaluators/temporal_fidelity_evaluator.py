from src.eval.evaluators.evaluator import Evaluator
import torch
import os
from typing import List

class TemporalFidelityEvaluator(Evaluator):
    def __init__(self, eval_args, output_dir: str) -> None:
        super().__init__(eval_args, output_dir)         
        self.eval_dir = os.path.join(self.eval_dir, "temporal_fidelity/")

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor):
        raise NotImplementedError

    def _post_processing(self):
        raise NotImplementedError