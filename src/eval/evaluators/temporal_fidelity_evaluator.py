from src.eval.evaluators.evaluator import Evaluator
import torch
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.eval.fidelity.methods_temporal_fidelity import similarity_auto_correlations, similarity_event_distributions, similarity_temporal_distances

class TemporalFidelityEvaluator(Evaluator):
    def __init__(self, eval_args, output_dir: str) -> None:
        super().__init__(eval_args, output_dir)         
        self.eval_dir = os.path.join(self.eval_dir, "temporal_fidelity/")

        self.total_results_event_distributions = []
        self.total_results_temporal_distances = []
        self.total_results_autocorrelations = []

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor):
        self.total_results_event_distributions.append(similarity_event_distributions(self.eval_args.real_data.sequences, syndata))
        self.total_results_temporal_distances.append(similarity_temporal_distances(self.eval_args.real_data.sequences, syndata))
        self.total_results_autocorrelations.append(similarity_auto_correlations(self.eval_args.real_data.sequences, syndata, self.eval_args.columns))

    def _post_processing(self):
        print("Post-processing temporal fidelity results")
        unzipped_distances_distributions = zip(*self.total_results_event_distributions)
        unzipped_temporal_distances = zip(*self.total_results_temporal_distances)
        unzipped_correlations = zip(*self.total_results_autocorrelations)
        distances_event_distributions, distances_per_variable = map(list, unzipped_distances_distributions)
        sim_scores_distances, sim_matrices_distances = map(list, unzipped_temporal_distances)
        sim_scores_correlations, sim_matrices_correlations = map(list, unzipped_correlations)

        # Save non-averaged results to dataframe
        non_avg_scores_df = pd.DataFrame()
        non_avg_scores_df['Similarity Score event distributions'] = pd.Series(distances_event_distributions)
        non_avg_scores_df['Temporal Distance Score'] = pd.Series(sim_scores_distances)
        non_avg_scores_df['Autocorrelation Score'] = pd.Series(sim_scores_correlations)

        # Calculate averages
        avg_matrix_distances = sum(sim_matrices_distances) / len(sim_matrices_distances)
        avg_matrix_correlations = sum(sim_matrices_correlations) / len(sim_matrices_correlations)

        # Create DataFrame for avg_statistics_per_variable
        avg_distances_event_distributions_per_variable = pd.DataFrame(distances_per_variable)

        # Convert avg_statistics_per_variable table to LaTeX and save to file
        latex_table = avg_distances_event_distributions_per_variable.to_latex(index=False)
        with open(f'{self.eval_dir}/{self.eval_args.model_type}_distances_temp_distributions_per_var.tex', 'w') as f:
            f.write(latex_table)

        # Convert avg_statistics_per_variable table to Markdown and save to file
        markdown_table = avg_distances_event_distributions_per_variable.to_markdown(index=False)
        with open(f'{self.eval_dir}/{self.eval_args.model_type}_distances_temp_distributions_per_var.md', 'w') as f:
            f.write(markdown_table)

        # Visualize avg temporal distance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_matrix_distances, annot=True, cmap='coolwarm')
        plt.title('Average Correlation Matrix')
        plot_path = f'{self.eval_dir}/{self.eval_args.model_type}_heatmap_temporal_distances.png'
        plt.savefig(plot_path)
        plt.close()

        # Visualize avg autocorrelation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_matrix_correlations, annot=True, cmap='coolwarm')
        plt.title('Average Correlation Matrix')
        plot_path = f'{self.eval_dir}/{self.eval_args.model_type}_heatmap_autocorrelations.png'
        plt.savefig(plot_path)
        plt.close()

        return non_avg_scores_df
