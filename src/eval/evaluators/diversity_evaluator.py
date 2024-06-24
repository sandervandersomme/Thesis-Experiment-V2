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
        self.total_intra_real_sequences = []
        self.total_intra_syn_sequences = []
        self.total_inter_sequences = []

        self.total_rel_diversity_events = []
        self.total_rel_coverage_events = []
        self.total_intra_real_events = []
        self.total_intra_syn_events = []
        self.total_inter_events = []

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor):
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]

        rel_coverage_sequences, rel_diversity_sequences, intra_d_syn_sequences, intra_d_real_sequences, inter_d_sequences = calculate_diversity_scores(
            train_data, syndata, self.eval_args.n_components, self.eval_args.n_neighbors_diversity, self.eval_args.coverage_factor, "sequences")

        rel_coverage_events, rel_diversity_events, intra_d_syn_events, intra_d_real_events, inter_d_events = calculate_diversity_scores(
            train_data, syndata, self.eval_args.n_components, self.eval_args.n_neighbors_diversity, self.eval_args.coverage_factor, "events")
        
        self.total_rel_diversity_sequences.append(rel_diversity_sequences)
        self.total_rel_coverage_sequences.append(rel_coverage_sequences)
        self.total_intra_real_sequences.append(intra_d_real_sequences)
        self.total_intra_syn_sequences.append(intra_d_syn_sequences)
        self.total_inter_sequences.append(inter_d_sequences)

        self.total_rel_diversity_events.append(rel_diversity_events)
        self.total_rel_coverage_events.append(rel_coverage_events)
        self.total_intra_real_events.append(intra_d_real_events)
        self.total_intra_syn_events.append(intra_d_syn_events)
        self.total_inter_events.append(inter_d_events)


    def _post_processing(self):
        # Save non-averaged results to dataframe
        non_avg_scores_df = pd.DataFrame()
        non_avg_scores_df['Relative Sequence Diversity'] = pd.Series(self.total_rel_diversity_sequences)
        non_avg_scores_df['Relative Sequence Coverage'] = pd.Series(self.total_rel_coverage_sequences)
        non_avg_scores_df['Intra Distance Real Sequences'] = pd.Series(self.total_intra_real_sequences)
        non_avg_scores_df['Intra Distance Syn Sequences'] = pd.Series(self.total_intra_syn_sequences)
        non_avg_scores_df['Inter Distance Sequences'] = pd.Series(self.total_inter_sequences)

        non_avg_scores_df['Relative Event Diversity'] = pd.Series(self.total_rel_diversity_events)
        non_avg_scores_df['Relative Event Coverage'] = pd.Series(self.total_rel_coverage_events)
        non_avg_scores_df['Intra Distance Real Events'] = pd.Series(self.total_intra_real_events)
        non_avg_scores_df['Intra Distance Syn Events'] = pd.Series(self.total_intra_syn_events)
        non_avg_scores_df['Inter Distance Events'] = pd.Series(self.total_inter_events)



        return non_avg_scores_df
