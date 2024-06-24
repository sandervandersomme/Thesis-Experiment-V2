from src.eval.evaluators.evaluator import Evaluator
import os
import torch
from typing import List
from src.eval.diversity.methods_diversity import calculate_diversity_scores
import pandas as pd

class DiversityEvaluator(Evaluator):
    def __init__(self, eval_args, output_dir: str):
        super().__init__(eval_args, output_dir)         
        self.eval_dir = os.path.join(self.eval_dir, "diversity/")

        self.total_rel_diversity_sequences = []
        self.total_rel_coverage_sequences = []
        self.total_abs_coverage_sequences = []

        self.total_rel_diversity_events = []
        self.total_rel_coverage_events = []
        self.total_abs_coverage_events = []

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor):
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]
        abs_coverage_sequences, rel_coverage_sequences, rel_diversity_sequences = calculate_diversity_scores(
            train_data, syndata, self.eval_args.n_components, self.eval_args.n_neighbors_diversity, "sequences")

        abs_coverage_events, rel_coverage_events, rel_diversity_events = calculate_diversity_scores(
            train_data, syndata, self.eval_args.n_components, self.eval_args.n_neighbors_diversity, "events")
        
        self.total_rel_diversity_sequences.append(rel_diversity_sequences)
        self.total_rel_coverage_sequences.append(rel_coverage_sequences)
        self.total_abs_coverage_sequences.append(abs_coverage_sequences)

        self.total_rel_diversity_events.append(rel_diversity_events)
        self.total_rel_coverage_events.append(rel_coverage_events)
        self.total_abs_coverage_events.append(abs_coverage_events)

    def _post_processing(self):
        # Save non-averaged results to dataframe
        non_avg_scores_df = pd.DataFrame()
        non_avg_scores_df['Relative Sequence Diversity'] = pd.Series(self.total_rel_diversity_sequences)
        non_avg_scores_df['Relative Sequence Coverage'] = pd.Series(self.total_rel_coverage_sequences)
        non_avg_scores_df['Absolute Sequence Coverage'] = pd.Series(self.total_abs_coverage_sequences)
        non_avg_scores_df['Relative Event Diversity'] = pd.Series(self.total_rel_diversity_events)
        non_avg_scores_df['Relative Event Coverage'] = pd.Series(self.total_rel_coverage_events)
        non_avg_scores_df['Absolute Event Coverage'] = pd.Series(self.total_abs_coverage_events)

        return non_avg_scores_df
