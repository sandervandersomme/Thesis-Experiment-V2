import torch

from typing import List

from eval.metric import Metric

class Evaluator:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.results = {}

    def evaluate(self, real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str], task: str):

        assert real_data.shape[1] == synthetic_data.shape[1]
        assert real_data.shape[2] == synthetic_data.shape[2]
        assert real_data.shape[2] == len(columns)

        for metric in self.metrics:
            self.results[metric.name] = metric.evaluate(real_data=real_data, synthetic_data=synthetic_data, columns=columns, task=task)
        return self.results
