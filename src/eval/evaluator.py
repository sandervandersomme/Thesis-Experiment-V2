import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.models.models import GenModel, task_to_model
from src.training.hyperparameters import load_default_params, load_optimal_params, add_shape_to_params
from src.utils import convert_numpy_types
from src.eval.similarity.methods_fidelity import avg_diff_statistics, wasserstein_distance, similarity_of_correlations
from src.eval.similarity.methods_temporal_fidelity import similarity_event_distributions, similarity_temporal_distances, similarity_temporal_dependencies
from src.eval.diversity.methods_diversity import calculate_diversity_scores
from src.eval.utility.methods_utility import run_downstream_task
from src.eval.privacy.methods_privacy import calculate_direct_matches, perform_aia, mia_blackbox_attack, mia_whitebox_attack, reidentification_risk
from typing import List

class Evaluator:
    def __init__(self, criteria: List[str], real_data: Dataset, train_indices: torch.Tensor, test_indices: torch.Tensor, output_dir: str, param_dir: str, args):
        self.criteria = criteria
        self.real_data = real_data
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.args = args
        self.results_df = pd.DataFrame()
        self.output_path = output_dir
        self.param_dir = param_dir

    def evaluate(self, model: GenModel, syndata: torch.Tensor, model_name: str, model_id: int, syndata_id: int):
        self.syndata = syndata
        self.model = model
        self.model_info = {
            "model_name": model_name,
            "model_id": model_id,
            "syndata_id": syndata_id
        }
        self.filename = f"results_{self.args.dataset}-{model_name}-{model_id}-{syndata_id}"
        self.model_scores = {}

        if "all" in self.criteria:
            self.evaluate_all()
        else:
            if "fidelity" in self.criteria:
                self.evaluate_fidelity()
            if "temporal_fidelity" in self.criteria:
                self.evaluate_temporal_fidelity()
            if "diversity" in self.criteria:
                self.evaluate_diversity()
            if "utility" in self.criteria:
                self.evaluate_utility()
            if "privacy" in self.criteria:
                self.evaluate_privacy()
        
        self.save_json(self.model_scores, self.filename)
        self.update_results_df()

    def evaluate_all(self):
        self.evaluate_fidelity()
        self.evaluate_temporal_fidelity()
        self.evaluate_diversity()
        self.evaluate_utility()
        self.evaluate_privacy()

    def save_json(self, results, filename):
        with open(f'{self.output_path}{filename}.json', 'w') as file:
            json.dump(results, file, indent=4, default=convert_numpy_types)

    def evaluate_fidelity(self):
        print("Evaluating fidelity..")
        self.model_scores.update(
            **avg_diff_statistics(self.real_data.sequences, self.syndata, self.real_data.columns), 
            **similarity_of_correlations(self.real_data.sequences, self.syndata, self.output_path), 
            **wasserstein_distance(self.real_data.sequences, self.syndata, self.real_data.columns, self.output_path))

    def evaluate_temporal_fidelity(self):
        print("Evaluating temporal fidelity..")

        self.model_scores.update(
            **similarity_event_distributions(self.real_data.sequences, self.syndata, self.output_path), 
            **similarity_temporal_distances(self.real_data.sequences, self.syndata, self.output_path), 
            **similarity_temporal_dependencies(self.real_data.sequences, self.syndata, self.real_data.columns, self.output_path))

    def evaluate_utility(self):
        print("Evaluating utility..")
        train_sequences = self.real_data[self.train_indices]
        test_sequences = self.real_data[self.test_indices]
        for task in self.args.tasks:
            downstream_model_type = task_to_model(task)

            if self.args.flag_default_params: hyperparams = load_default_params(downstream_model_type)
            else: hyperparams = load_optimal_params(self.param_dir, f"{self.args.dataset}-{self.model_info["model_name"]}-{self.args.seed}")

            self.model_scores.update(
                run_downstream_task(task, self.args.dataset, self.syndata, train_sequences, test_sequences, self.real_data.columns, hyperparams, self.args.epochs, self.args.val_split_size, self.args.seed)
            )

    def evaluate_diversity(self):
        print("Evaluating diversity..")

        self.model_scores.update(
            calculate_diversity_scores(self.real_data.sequences, self.syndata, self.args.n_components, self.args.n_neighbors_diversity))

    def evaluate_privacy(self):
        print("Evaluating privacy..")

        train_sequences = self.real_data[self.train_indices]
        test_sequences = self.real_data[self.test_indices]
        self.model_scores.update(
            **mia_blackbox_attack(self.syndata, train_sequences, test_sequences, self.model, self.args.mia_threshold, self.args.epochs),
            **mia_whitebox_attack(train_sequences, test_sequences, self.model, self.args.mia_threshold),
            **calculate_direct_matches(self.syndata, train_sequences, self.args.matching_threshold),
            **reidentification_risk(train_sequences, self.syndata, self.args.reid_threshold),
            **perform_aia(self.syndata, train_sequences, self.args.n_neighbors_privacy, self.args.aia_threshold, self.args.num_hidden_attributes))
        
    def update_results_df(self):
        filtered_results = {k: v for k, v in self.model_scores.items() if not isinstance(v, (dict, list, tuple, np.ndarray))}
        df = pd.DataFrame(filtered_results, index=[0])
        df['model_name'] = self.model_info['model_name']
        df['model_id'] = self.model_info['model_id']
        df['syndata_id'] = self.model_info['syndata_id']
        self.results_df = pd.concat([self.results_df, df], ignore_index=True)

    def save_results_to_csv(self):
        self.results_df.to_csv(f'{self.output_path}results_all_datasets.csv', index=False)

    def save_averages(self):
        average_results = self.results_df.groupby('model_name').mean()
        average_results.to_csv(f"{self.output_path}averages_models.csv", index=False)