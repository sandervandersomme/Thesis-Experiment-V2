from typing import List
from src.utils import get_filenames_of_models, get_filenames_of_syndatasets, load_model

# Import evaluators
from src.eval.evaluators.evaluator import Evaluator
from src.eval.evaluators.fidelity_evaluator import FidelityEvaluator
from src.eval.evaluators.temporal_fidelity_evaluator import TemporalFidelityEvaluator
from src.eval.evaluators.diversity_evaluator import DiversityEvaluator
from src.eval.evaluators.utility_evaluator import UtilityEvaluator
from src.eval.evaluators.privacy_evaluator import PrivacyEvaluator

class Collector():
    def __init__(self, criteria: List[str], models: List[str], num_instances: int, num_datasets: int, args, output_dir: str) -> None:
        self.criteria = criteria
        self.models = models
        self.num_instances = num_instances
        self.num_datasets = num_datasets

        self.output_dir = output_dir
        self.args = args

        self.results_full = None
        self.results_average = None

    def collect_results(self):
        for model_type in self.args.models:
            model_files = get_filenames_of_models(model_type, self.num_instances)

            self.collect_model_results(model_type, model_files)

    def collect_model_results(self, model: str, model_files: List[str]):
        print(f"Start evaluating model {model}..")
        # Create new evaluators for model
        self.evaluators = create_evaluators(self.criteria, self.args, self.output_dir)

        # Get files of datasets
        filenames_datasets = get_filenames_of_syndatasets(model, self.num_instances, self.num_datasets)
        
        # Use evaluators for evaluation of the datasets
        for evaluator in self.evaluators:
            evaluator.evaluate(filenames_datasets)


def create_evaluators(criteria: List[str], args, output_dir) -> List[Evaluator]:
    if criteria == "all":
        return [
            FidelityEvaluator(args, output_dir)
        ]
    else: 
        for criterion in criteria:
            evaluators = []
            if criterion == "fidelity":
                evaluators.append(FidelityEvaluator(args, output_dir))
            if criterion == "temporal":
                raise NotImplementedError
            if criterion == "diversity":
                raise NotImplementedError
            if criterion == "utility":
                raise NotImplementedError
            if criterion == "privacy":
                raise NotImplementedError
    return evaluators
