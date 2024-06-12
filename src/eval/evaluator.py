import pandas as pd
import torch
import inspect
import argparse
import os

from src.eval.methods import similarity_methods, diversity_methods, privacy_methods, utility_methods, all_methods
from src.models.models import GenModel
from src.utilities.utils import convert_numpy_types

import pandas as pd
from typing import List, Callable
import json

class Evaluator:
    def __init__(self, train_data: torch.Tensor, test_data: torch.Tensor, name_dataset: str, methods: List[Callable], output_path: str, **kwargs):
        self.results = pd.DataFrame()
        self.name_dataset = name_dataset
        self.output_path = output_path
        self.methods = methods
        self.kwargs = kwargs
        self.kwargs.update({
            "train_data": train_data,
            "test_data": test_data
        })

        self.check_params_is_complete()

    def evaluate_dataset(self, syndata: torch.Tensor, model: GenModel):
        graph_path = f"{self.output_path}graphs/{self.name_dataset}-{model.NAME}/"
        os.makedirs(f"{graph_path}sim_ts_corrs/", exist_ok=True)
        self.kwargs.update({
            "syndata": syndata,
            "model" : model,
            "graph_path": graph_path
        })

        scores = {"Name dataset": self.name_dataset}

        for method in self.methods:
            eval_results = self.evaluate_method(method)
            scores.update(eval_results)

        output_file_path = f"outputs/experiments/1/scores-{self.kwargs["model"].NAME}{self.name_dataset}.json"
        with open(output_file_path, 'w') as f:
            json.dump(scores, f, default=convert_numpy_types)
    
    def evaluate_method(self, method: Callable):
        # Dynamically set the arguments and run the evaluation methods
        required_params = inspect.signature(method).parameters
        method_args = {param: self.kwargs[param] for param in required_params}
        return method(**method_args)

    def get_results(self): return self.results
    def to_latex(self): return self.results.to_latex(index=False)
    def save_results(self, output_path): self.results.to_csv(output_path)

    def check_params_is_complete(self): 
        # Get all params of all methods without syndata and model
        all_required_params = set([param for method in self.methods for param in inspect.signature(method).parameters]).difference(["syndata", "model", "graph_path"])
        assert all_required_params.issubset(self.kwargs.keys()), f"Missing arguments: {all_required_params.difference(self.kwargs.keys())}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Privacy arguments
    parser.add_argument('--k', type=int, help="Number of neighbors in knn", default=1)
    parser.add_argument('--aia_threshold', type=float, help="", default=0.8)
    parser.add_argument('--mia_threshold', type=float, help="", default=0.8)
    parser.add_argument('--reid_threshold', type=float, help="", default=0.8)
    parser.add_argument('--matching_threshold', type=float, help="", default=0.8)
    parser.add_argument('--num_disclosed_attributes', type=int, help="Number of disclosed attributes in attribute inference attack", default=3)
    
    # Diversity arguments
    parser.add_argument('--n_components', type=int, help="Number of componenents in pca", default=10)
    parser.add_argument('--n_neighbors', type=int, help="Number of neighbors in knn", default=5)
    parser.add_argument('--reshape_method', type=str, help="How to reshape the data?", choices=['sequences', 'events'], default=0)

    method_args = vars(parser.parse_args())

    method_args.update({
            "train_data": None,
            "test_data": None,
            "columns": None,
            "model": None
    })

    eval = Evaluator(diversity_methods, **method_args)
    eval.evaluate(f"outputs/syndata/1")