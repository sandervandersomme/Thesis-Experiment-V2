from src.eval.evaluators.evaluator import Evaluator
from src.eval.fidelity.methods_fidelity import similarity_of_statistics, wasserstein_distance, similarity_of_correlations
from src.utils import save_df_to_csv, save_matrix_to_np
import pandas as pd
import os
import torch

class FidelityEvaluator(Evaluator):
    def __init__(self, syndata: torch.Tensor, eval_args, output_dir: str) -> None:
        super().__init__(syndata, eval_args, output_dir)

    def setup_paths(self):
        # Set root directory
        self.eval_dir = os.path.join(self.eval_dir, f"fidelity/{self.eval_args.model_type}")

        # Set filename
        self.filename = f"{self.eval_args.model_id}_{self.eval_args.syndata_id}"

        # Set paths to dirs
        self.path_scores = os.path.join(self.eval_dir, f"scores/")
        self.path_varstats = os.path.join(self.eval_dir, f"stats_per_var/")
        self.path_corrs = os.path.join(self.eval_dir, f"correlations/")

    def setup_folders(self):
        # Create dirs
        os.makedirs(self.path_scores, exist_ok=True)
        os.makedirs(self.path_varstats, exist_ok=True)
        os.makedirs(self.path_corrs, exist_ok=True)
        
    def evaluate(self):
        print(f"Start fidelity evaluation of model {self.eval_args.model_type}..")

        # Create dataframe
        scores = pd.DataFrame()
        scores = pd.concat([scores, self.evaluate_basic_statistics().to_frame().T])
        scores["Sim. Score Correlations"] = self.evaluate_correlations()
        scores["Wasserstein Distance"] = self.evaluate_distances()

        # Add identifier
        scores["Model"] = self.eval_args.model_type

        # Store metric scores
        path = os.path.join(self.path_scores, self.filename)
        save_df_to_csv(scores, path)
        
    def evaluate_basic_statistics(self):
        print(f"Evaluating basic statistics..")
        similarity_scores_per_variable = similarity_of_statistics(self.eval_args.real_data.sequences, self.syndata, self.eval_args.columns)

        # Store all similarities of all variables of current dataset
        path = os.path.join(self.path_varstats, self.filename)
        save_df_to_csv(similarity_scores_per_variable, path)

        # Compute average similarity scores
        similarity_scores = similarity_scores_per_variable.mean(axis=0)
        return similarity_scores

    def evaluate_correlations(self):
        print(f"Evaluating correlations..")
        similarity_score, matrix = similarity_of_correlations(self.eval_args.real_data.sequences, self.syndata)
        
        # Storing matrices
        path = os.path.join(self.path_corrs, self.filename)
        save_matrix_to_np(matrix, path)

        return similarity_score
    
    def evaluate_distances(self):
        print(f"Evaluating distance..")

        distance = wasserstein_distance(self.eval_args.real_data.sequences, self.syndata, self.eval_args.columns)
        return distance

# TODO: Visualise matrices
# TODO: Take average table of stats per var across all models