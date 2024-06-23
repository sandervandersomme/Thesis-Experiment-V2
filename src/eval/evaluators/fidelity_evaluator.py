import torch
from src.eval.evaluators.evaluator import Evaluator
from src.eval.fidelity.methods_fidelity import similarity_of_statistics, wasserstein_distance, similarity_of_correlations
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FidelityEvaluator(Evaluator):
    def __init__(self, eval_args, output_dir: str) -> None:
        super().__init__(eval_args, output_dir)
         
        self.eval_dir = os.path.join(self.eval_dir, "fidelity/")
        self.all_statistics = []
        self.all_correlations = []
        self.all_distances = []

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor):
        results_statistics = similarity_of_statistics(self.eval_args.real_data.sequences, syndata, self.eval_args.columns)
        results_correlations = similarity_of_correlations(self.eval_args.real_data.sequences, syndata)
        results_distances = wasserstein_distance(self.eval_args.real_data.sequences, syndata, self.eval_args.columns)

        self.all_statistics.append(results_statistics)
        self.all_correlations.append(results_correlations)
        self.all_distances.append(results_distances)

    def _post_processing(self):
        unzipped_statistics = zip(*self.all_statistics)
        statistics, statistics_per_variable = map(list, unzipped_statistics)
        unzipped_correlations = zip(*self.all_correlations)
        correlation_scores, correlation_matrices = map(list, unzipped_correlations)

        # Save non-averaged results to dataframe
        non_avg_scores_df = pd.DataFrame(statistics)
        non_avg_scores_df['Correlation Score'] = pd.Series(correlation_scores)
        non_avg_scores_df['Distance Score'] = pd.Series(self.all_distances)

        # Calculate averages
        avg_statistic_scores = sum(statistics)/len(statistics)
        avg_statistics_per_variable = sum(statistics_per_variable)/len(statistics_per_variable)
        avg_correlation_score = sum(correlation_scores)/len(correlation_scores)
        avg_correlation_matrix = sum(correlation_matrices)/len(correlation_matrices)
        avg_distance = sum(self.all_distances)/len(self.all_distances)

        # Save averaged results to dataframe
        avg_scores_df = pd.DataFrame({
            'Statistic Mean': [avg_statistic_scores.iloc[0]],
            'Statistic Std': [avg_statistic_scores.iloc[1]],
            'Statistic Median': [avg_statistic_scores.iloc[2]],
            'Statistic Var': [avg_statistic_scores.iloc[3]],
            'Correlation Score': [avg_correlation_score],
            'Distance Score': [avg_distance]
        })
        
        # Create DataFrame for avg_statistics_per_variable
        avg_statistics_df = pd.DataFrame(avg_statistics_per_variable)

        # Convert avg_statistics_per_variable table to LaTeX and save to file
        latex_table = avg_statistics_df.to_latex(index=False)
        with open(f'{self.eval_dir}/{self.eval_args.model_type}_non_avg_statistics.tex', 'w') as f:
            f.write(latex_table)

        # Convert avg_statistics_per_variable table to Markdown and save to file
        markdown_table = avg_statistics_df.to_markdown(index=False)
        with open(f'{self.eval_dir}/{self.eval_args.model_type}_non_avg_statistics.md', 'w') as f:
            f.write(markdown_table)

        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Average Correlation Matrix')
        plot_path = f'{self.eval_dir}/{self.eval_args.model_type}_heatmap_correlations.png'
        plt.savefig(plot_path)
        plt.close()

        return avg_scores_df, non_avg_scores_df