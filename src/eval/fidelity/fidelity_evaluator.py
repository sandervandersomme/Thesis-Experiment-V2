import torch
from src.models.gen_model import GenModel
from src.eval.newevaluator import Evaluator
from src.eval.fidelity.methods_fidelity import similarity_of_statistics, wasserstein_distance, similarity_of_correlations
import os
import numpy as np
import pandas as pd 

class FidelityEvaluator(Evaluator):
    def __init__(self, eval_args, eval_dir: str) -> None:
        eval_dir = os.path.join(eval_dir, "fidelity/")
        super().__init__(eval_args, eval_dir)
        self.num_datasets = 0
         
        self.total_sim_matrix_stats = None
        self.total_stats_sim_scores = None
        self.total_correlations = None
        self.total_sim_score = 0.0
        self.total_w_distance = 0.0

        self.results_df = pd.DataFrame()

    def evaluate_dataset(self, syndata: torch.Tensor, dataset_id: int, model: GenModel, model_id: int):
        sim_matrix_stats, stat_sim_scores = similarity_of_statistics(self.args.real_data, syndata, self.args.columns), 
        correlations, similarity_score = similarity_of_correlations(self.args.real_data, syndata), 
        w_distance = wasserstein_distance(self.args.real_data, self.syndata, self.args.columns)

        self.total_sim_matrix_stats = sim_matrix_stats if self.total_sim_matrix_stats is None else self.total_sim_matrix_stats + sim_matrix_stats
        self.total_stats_sim_scores = stat_sim_scores if self.total_stats_sim_scores is None else self.total_stats_sim_scores + stat_sim_scores
        self.total_correlations = sim_matrix_stats if self.total_correlations is None else self.total_correlations + correlations
        self.total_sim_score += similarity_score
        self.total_w_distance += w_distance

        self.num_datasets += 1

        # Add results to DataFrame
        result_row = {
            'model': model.NAME,
            'model_id': model_id,
            'dataset_id': dataset_id,
            **dict(stat_sim_scores),
            'similarity_score': similarity_score,
            'w_distance': w_distance,
        }

        self.results_df = self.results_df.append(result_row, ignore_index=True)

    def calculate_averages(self):
        avg_sim_matrix_stats = (self.total_sim_matrix_stats / self.num_datasets).mean()
        avg_sim_scores_stats = (self.total_stats_sim_scores / self.num_datasets).mean()
        avg_total_correlations = self.total_correlations / self.num_datasets
        avg_sim_score = self.total_sim_score / self.num_datasets
        avg_w_distance = self.total_w_distance / self.num_datasets



    # TODO add real_data to eval_args, add columns to args
    # TODO rewrite wasserstein distance such that it is a similarity metric
    # TODO Rewrite basic statistic section to be similarities

