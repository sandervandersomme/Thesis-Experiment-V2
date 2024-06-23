import torch
from src.eval.evaluators.evaluator import Evaluator
from src.eval.fidelity.methods_fidelity import similarity_of_statistics, wasserstein_distance, similarity_of_correlations
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FidelityEvaluator(Evaluator):
    def __init__(self, eval_args, output_dir: str) -> None:
        super().__init__(eval_args, output_dir)
         
        self.all_statistics = []
        self.all_correlations = []
        self.all_distances = []

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor, dataset_id: int):
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

        # Calculate averages
        avg_statistic_scores = sum(statistics)/len(statistics)
        avg_statistics_per_variable = sum(statistics_per_variable)/len(statistics_per_variable)
        avg_correlation_score = sum(correlation_scores)/len(correlation_scores)
        avg_correlation_matrix = sum(correlation_matrices)/len(correlation_matrices)
        avg_distance = sum(self.all_distances)/len(self.all_distances)

        # Save to dataframe
        avg_results = pd.DataFrame({
            'Statistic Mean': [avg_statistic_scores.iloc[0]],
            'Statistic Std': [avg_statistic_scores.iloc[1]],
            'Statistic Median': [avg_statistic_scores.iloc[2]],
            'Statistic Var': [avg_statistic_scores.iloc[3]],
            'Correlation Score': [avg_correlation_score],
            'Distance Score': [avg_distance]
        })
        print(avg_results)

        # # Visualize correlation matrix
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(avg_correlation_matrix, annot=True, cmap='coolwarm')
        # plt.title('Average Correlation Matrix')
        # plt.show()

        # # Print table of statistics
        # print(df)

        # # Save dataframe to CSV
        # df.to_csv(f'{self.output_dir}/evaluation_results.csv', index=False)

        # return df
