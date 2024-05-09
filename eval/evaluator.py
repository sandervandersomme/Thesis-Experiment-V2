from typing import List

from metric import Metric

class Evaluator:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.results = {}

    def evaluate(self, data, columns: List[str]):
        for metric in self.metrics:
            self.results[metric.name] = metric.evaluate(data, columns)