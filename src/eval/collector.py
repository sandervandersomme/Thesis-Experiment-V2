from typing import List

from src.eval.diversity.diversity_evaluator import DiversityEvaluator
from src.eval.privacy.privacy_evaluator import PrivacyEvaluator
from src.eval.similarity.fidelity_evaluator import FidelityEvaluator
from src.eval.similarity.temporal_fidelity_evaluator import TemporalFidelityEvaluator
from src.eval.utility.utility_evaluator import UtilityEvaluator
from src.utils import get_filenames_of_models, get_filenames_of_syndatasets, load_model, load_dataset
from src.eval.newevaluator import Evaluator

class Collector():
    def __init__(self, models, criteria, eval_dir, eval_args, num_instances, num_datasets) -> None:
        self.models = models
        self.criteria = criteria
        self.eval_dir = eval_dir
        self.eval_args = eval_args
        self.num_instances = num_instances
        self.num_datasets = num_datasets

        self.results_full = None
        self.results_average = None

    def collect_results(self):
        for model_type in self.models:
            self.model_type = model_type
            self.filenames_datasets = get_filenames_of_syndatasets(model_type, self.num_instances, self.num_datasets)
            self.filenames_models = get_filenames_of_models(model_type, self.num_instances)
            self.evaluate_criteria()

    def _evaluate_fidelity(self):
        # Create evaluator
        evaluator = FidelityEvaluator(self.eval_args, self.eval_dir)
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
        for criterion in self.criteria:
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
                evaluator.evaluate_dataset(syndata, syndata_id, model, model_id)

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

