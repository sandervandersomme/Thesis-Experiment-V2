from src.eval.evaluators.evaluator import Evaluator
import torch
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import save_df_to_csv, save_matrix_to_np

from src.eval.fidelity.methods_temporal_fidelity import similarity_auto_correlations, similarity_event_distributions, similarity_temporal_distances

class TemporalFidelityEvaluator(Evaluator):
    def __init__(self, syndata: torch.Tensor, eval_args, output_dir: str) -> None:
        super().__init__(syndata, eval_args, output_dir)         

    def setup_paths(self):
        # Set root directory
        self.eval_dir = os.path.join(self.eval_dir, f"temporal/{self.eval_args.model_type}")

        # Set filename
        self.filename = f"{self.eval_args.model_id}_{self.eval_args.syndata_id}"

        # Set paths to dirs
        self.path_scores = os.path.join(self.eval_dir, f"scores/")
        self.path_distributions = os.path.join(self.eval_dir, f"sim_event_distributions/")
        self.path_distances = os.path.join(self.eval_dir, f"sim_temporal_distances/")
        self.path_autocorr = os.path.join(self.eval_dir, f"sim_autocorrelations/")

    def setup_folders(self):
        # Create dirs
        os.makedirs(self.path_scores, exist_ok=True)
        os.makedirs(self.path_distributions, exist_ok=True)
        os.makedirs(self.path_distances, exist_ok=True)
        os.makedirs(self.path_autocorr, exist_ok=True)
        
    def evaluate(self):
        print(f"Start temporal fidelity evaluation of model {self.eval_args.model_type}..")

        # Create dataframe
        scores = pd.DataFrame()
        scores["Model"] = pd.Series(self.eval_args.model_type)
        scores["Sim. Score Event Distributions"] = self.evaluate_event_distributions()
        scores["Sim. Score Temporal Distances"] = self.evaluate_temporal_distances()
        scores["Sim. Score Autocorrelations"] = self.evaluate_auto_correlations()

        # # Store metric scores
        path = os.path.join(self.path_scores, self.filename)
        save_df_to_csv(scores, path)
        
    def evaluate_event_distributions(self):
        print("Evaluating event distributions")
        avg_distance, var_distances = similarity_event_distributions(self.eval_args.real_data.sequences, self.syndata)

        var_distances_df = pd.Series(var_distances).to_frame().T
        path = os.path.join(self.path_distributions, self.filename)
        save_df_to_csv(var_distances_df, path)

        return pd.Series(avg_distance)
    
    def evaluate_temporal_distances(self):
        print(f"Evaluating temporal distances..")
        sim_score, matrix = similarity_temporal_distances(self.eval_args.real_data.sequences, self.syndata)
        
        # Storing matrices
        path = os.path.join(self.path_distances, self.filename)
        save_matrix_to_np(matrix, path)

        return pd.Series(sim_score)

    def evaluate_auto_correlations(self):
        print(f"Evaluating autocorrelations..")
        sim_score, matrix = similarity_auto_correlations(self.eval_args.real_data.sequences, self.syndata, self.eval_args.columns)
        
        # Storing matrices
        path = os.path.join(self.path_autocorr, self.filename)
        save_matrix_to_np(matrix, path)

        return pd.Series(sim_score)
