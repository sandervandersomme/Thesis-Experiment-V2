from typing import List
from src.utils import load_dataset
import os

class Evaluator():
    def __init__(self, eval_args: str, eval_dir) -> None:
        self.eval_args = eval_args
        self.eval_dir = os.path.join(eval_dir, "eval/")
        self.syn_data_dir = os.path.join(eval_dir, "syndata/")

        # Collect results
        self.results = None
        self.average_results = None

    def evaluate(self, files: List[str]):
        for id, filename in enumerate(files):
            syn_dataset = load_dataset(self.syn_data_dir, filename)
            self._evaluate_dataset(syn_dataset, id)
        
        self._post_processing()

    def _evaluate_dataset(self):
        raise NotImplementedError
    
    def _post_processing(self):
        raise NotImplementedError
    
    def save_results():
        raise NotImplementedError

