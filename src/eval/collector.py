from typing import List
import os
import torch
from src.utils import load_model, load_syndata, save_df_to_csv

# Import evaluators
from src.eval.evaluators.evaluator import Evaluator
from src.eval.evaluators.fidelity_evaluator import FidelityEvaluator
from src.eval.evaluators.temporal_fidelity_evaluator import TemporalFidelityEvaluator
from src.eval.evaluators.diversity_evaluator import DiversityEvaluator
from src.eval.evaluators.utility_evaluator import UtilityEvaluator
from src.eval.evaluators.privacy_evaluator import PrivacyEvaluator

import pandas as pd

class Collector():
    def __init__(self, criteria: List[str], models: List[str], num_instances: int, num_datasets: int, args, output_dir: str) -> None:
        self.criteria = criteria
        self.models = models
        self.num_instances = num_instances
        self.num_datasets = num_datasets

        self.output_dir = output_dir
        self.eval_dir = os.path.join(output_dir, 'eval/')
        self.model_dir = os.path.join(output_dir, 'models/')
        self.syndata_dir = os.path.join(output_dir, 'syndata/')
        self.args = args

        self.results_full = pd.DataFrame()
        # self.results_average = pd.DataFrame()

        self.setup_folders()

    def setup_folders(self):
        for criterion in self.criteria:
            path = os.path.join(self.eval_dir, f"{criterion}/")
            os.makedirs(path, exist_ok=True)

    def collect_results(self):
        for model_type in self.args.models:
            self.args.model_type = model_type

            for model_id in range(self.args.num_instances):
                path = os.path.join(self.model_dir, f"{model_type}-{model_id}.pkl")
                self.args.model = load_model(path)
                self.args.model_id = model_id
                for syndata_id in range(self.args.num_instances):
                    path = os.path.join(self.syndata_dir, f"{model_type}-{model_id}-{syndata_id}.pt")
                    syndata = load_syndata(path)
                    self.args.syndata_id = syndata_id
                    for criterion in self.criteria:
                        evaluator = select_evaluator(criterion, syndata, self.args, self.output_dir)
                        evaluator.evaluate()


def select_evaluator(criterion: str, syndata: torch.Tensor, args, output_dir: str):
    if criterion == "fidelity":
        return FidelityEvaluator(syndata, args, output_dir)
    if criterion == "temporal":
        return TemporalFidelityEvaluator(syndata, args, output_dir)
    if criterion == "diversity":
        return DiversityEvaluator(syndata, args, output_dir)
    if criterion == "utility":
        return UtilityEvaluator(syndata, args, output_dir)
    if criterion == "privacy":
        return PrivacyEvaluator(syndata, args, output_dir)
    
    raise NotImplementedError