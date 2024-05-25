from eval.methods_diversity import diversity_methods
from eval.methods_privacy import privacy_methods
from eval.methods_similarity import similarity_methods
from eval.methods_utility import utility_methods

from typing import List, Callable
import torch

class Metric:
    def __init__(self, name: str, methods: List[Callable]):
        self.name = name
        self.methods = methods
        self.results = {}

    def evaluate(self, **kwargs):
        for method in self.methods:
            self.results[method.__name__] = method(**kwargs)
        return self.results

class Privacy(Metric):
    def __init__(self):
        super().__init__("privacy", privacy_methods)

class Similarity(Metric):
    def __init__(self):
        super().__init__("similarity", similarity_methods)

    # def evaluate(self, real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str]):
    #     return super().evaluate(real_data=real_data.numpy(), synthetic_data=synthetic_data.numpy(), columns=columns)
    
    def evaluate(self, **kwargs):
        real_data = kwargs["real_data"].numpy()
        synthetic_data = kwargs["synthetic_data"].numpy()
        columns = kwargs["columns"]
        return super().evaluate(real_data=real_data, synthetic_data=synthetic_data, columns=columns)

class Utility(Metric):
    def __init__(self):
        super().__init__("utility", utility_methods)

    def evaluate(self, **kwargs):
        # Train model with real data
        # train, test = 

        # Train model with synthetic data
        # Compare performance

        return super().evaluate()

class Diversity(Metric):
    def __init__(self):
        super().__init__("diversity", diversity_methods)



