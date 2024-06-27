from src.eval.evaluators.evaluator import Evaluator
import os
import torch
from typing import List
from src.eval.diversity.methods_diversity import calculate_diversity_scores
import pandas as pd

from src.utils import save_df_to_csv

class DiversityEvaluator(Evaluator):
    def __init__(self, syndata: torch.Tensor, eval_args, output_dir: str) -> None:
        super().__init__(syndata, eval_args, output_dir)         

    def setup_paths(self):
        # Set root directory
        self.eval_dir = os.path.join(self.eval_dir, f"diversity/{self.eval_args.model_type}")

        # Set filename
        self.filename = f"{self.eval_args.model_id}_{self.eval_args.syndata_id}"

        # Set paths to dirs
        self.path_scores = os.path.join(self.eval_dir, f"scores/")

    def setup_folders(self):
        # Create dirs
        os.makedirs(self.path_scores, exist_ok=True)

    def evaluate(self):
        print(f"Start diversity evaluation of model {self.eval_args.model_type}..")

        # Create dataframe
        scores = pd.DataFrame()
        scores["Model"] = pd.Series(self.eval_args.model_type)
        scores = pd.concat([scores, self.evaluate_sequence_diversity()], axis=1)
        scores = pd.concat([scores, self.evaluate_event_diversity()], axis=1)
        
        # # Store metric scores
        path = os.path.join(self.path_scores, self.filename)
        save_df_to_csv(scores, path)

    def evaluate_sequence_diversity(self):
        print(f"Evaluating sequence diversity..")

        # Prepare data
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]
        
        # Calculate metrics
        rel_coverage_sequences, rel_diversity_sequences, intra_d_syn_sequences, intra_d_real_sequences, inter_d_sequences = calculate_diversity_scores(
            train_data, self.syndata, self.eval_args.n_components, self.eval_args.n_neighbors_diversity, self.eval_args.coverage_factor, "events")

        df = pd.DataFrame()
        df["Rel. Diversity Seq"] = pd.Series(rel_diversity_sequences)
        df["Rel. Coverage Seq"] = pd.Series(rel_coverage_sequences)
        # df["Intra Dist. Real Seq"] = pd.Series(intra_d_real_sequences)
        # df["Intra Dist. Syn. Seq"] = pd.Series(intra_d_syn_sequences)
        # df["Inter Dist. Seq"] = pd.Series(inter_d_sequences)

        return df

    def evaluate_event_diversity(self):
        print(f"Evaluating event diversity..")

        # Prepare data
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]

        # Calculate metrics
        rel_coverage_events, rel_diversity_events, intra_d_syn_events, intra_d_real_events, inter_d_events = calculate_diversity_scores(
            train_data, self.syndata, self.eval_args.n_components, self.eval_args.n_neighbors_diversity, self.eval_args.coverage_factor, "events")

        df = pd.DataFrame()
        df["Rel. Diversity Events"] = pd.Series(rel_diversity_events)
        df["Rel. Coverage Events"] = pd.Series(rel_coverage_events)
        # df["Intra Dist. Real Seq"] = pd.Series(intra_d_real_events)
        # df["Intra Dist. Syn. Seq"] = pd.Series(intra_d_syn_events)
        # df["Inter Dist. Seq"] = pd.Series(inter_d_events)

        return df
