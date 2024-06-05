import torch
import inspect

from typing import List, Callable

class Evaluator:
    def __init__(self, methods: List[Callable]):
        self.methods = methods
        self.results = {}

    def evaluate(self, **kwargs):
        # Check if all arguments are 
        required_params = set()
        for method in self.methods:
            required_params.update(inspect.signature(method).parameters)

        # Assert that all required parameters are present in kwargs
        assert required_params.issubset(kwargs.keys()), f"Missing arguments: {required_params.difference(kwargs.keys())}"

        # Dynamically set the arguments and run the evaluation methods
        for method in self.methods:
            method_args = {param: kwargs[param] for param in inspect.signature(method).parameters}
            print(f"Running evaluation method: {method.__name__}")
            self.results[method.__name__] = method(**method_args)


if __name__ == "__main__":
    from src.eval.methods import privacy_methods
    from src.data.random_data import generate_random_data

    train_data = generate_random_data(20,5,20)
    test_data = generate_random_data(20,5,20)
    threshold = 0.8

    from src.models.rgan import RGAN
    from src.training.hyperparameters import get_default_params
    params = get_default_params("rgan", train_data.size())
    model = RGAN(**params)

    eval = Evaluator(privacy_methods)
    
    args = {
        "train_data": train_data,
        "test_data": test_data,
        "threshold": threshold,
        "model": model
    }

    eval.evaluate(**args)
    print(eval.results)