from typing import List
from torch.utils.data import Dataset
import torch

from src.eval.diversity.diversity_evaluator import DiversityEvaluator
from src.eval.privacy.privacy_evaluator import PrivacyEvaluator
from src.eval.fidelity.fidelity_evaluator import FidelityEvaluator
from src.eval.fidelity.temporal_fidelity_evaluator import TemporalFidelityEvaluator
from src.eval.utility.utility_evaluator import UtilityEvaluator
from src.utils import get_filenames_of_models, get_filenames_of_syndatasets, load_model
from src.eval.newevaluator import Evaluator

class Collector():
    def __init__(self, real_data: Dataset, train_indices: torch.Tensor, test_indices: torch.Tensor, args, eval_dir: str) -> None:
        self.real_data = real_data
        self.train_indices = train_indices
        self.test_indices = test_indices

        self.eval_dir = eval_dir
        self.args = args
        self.args

        self.results_full = None
        self.results_average = None

    def collect_results(self):
        for model_type in self.args.models:
            self.model_type = model_type
            self.filenames_datasets = get_filenames_of_syndatasets(model_type, self.args.num_instances, self.args.num_syn_datasets)
            self.filenames_models = get_filenames_of_models(model_type, self.args.num_instances)
            self.evaluate_criteria()

    def _evaluate_fidelity(self):
        # Create evaluator
        evaluator = FidelityEvaluator(self.args, self.eval_dir)
        results, average_results = self._evaluate(evaluator)
        # Add results
        raise NotImplementedError

    def _evaluate_temporal_fidelity(self):
        raise NotImplementedError

    def _evaluate_diversity(self):
        raise NotImplementedError

    def _evaluate_utility(self):
        raise NotImplementedError

    def _evaluate_privacy(self):
        raise NotImplementedError

    def evaluate_criteria(self):
        for criterion in self.args.criteria:
            if criterion == "all":
                self._evaluate_fidelity()
                self._evaluate_temporal_fidelity()
                self._evaluate_diversity()
                self._evaluate_utility()
                self._evaluate_privacy()
            else: 
                if criterion == "fidelity":
                    self._evaluate_fidelity()
                if criterion == "time":
                    self._evaluate_temporal_fidelity()
                if criterion == "diversity":
                    self._evaluate_diversity()
                if criterion == "utility":
                    self._evaluate_utility()
                if criterion == "time":
                    self._evaluate_privacy()

    def _evaluate(self, evaluator: Evaluator):
        # Loop through models
        for model_id, model_filename in enumerate(self.filenames_models):
            # Load model
            model = load_model(self.eval_dir, model_filename)

            # Loop through data
            for syndata_id, syndata_filename in enumerate(self.filenames_datasets):
                # Load data
                syndata = load_model(self.eval_dir, syndata_filename)
                
                # Evaluate data
                evaluator.evaluate_dataset(syndata, syndata_id, model, model_id, self.real_data, self.train_indices, self.test_indices)

        # Calculate averages
        # Visualise results
        # Return results, average results



if __name__ == "__main__":
    dir = "outputs/exp1/models/"
    models = ["rgan", "timegan", "rwgan"]
    criteria = ["all"]
    args = {}
    num_instances = 3
    num_datasets = 3

    collector = Collector(models, criteria, dir, args, num_instances, num_datasets)
    collector.collect_results()

