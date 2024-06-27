from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.utils import get_csv_files, save_df_to_csv, save_df_to_latex, save_df_to_markdown, get_npy_files

class PostProcessor():
    def __init__(self, criteria, models, dir: str) -> None:
        self.dir = dir
        self.dir_processed = os.path.join(self.dir, f"processed/")
        self.criteria = criteria
        self.models = models

        self.setup_folders()

    def setup_folders(self):
        dir = os.path.join(self.dir_processed, f"avgs_scores/")
        os.makedirs(dir, exist_ok=True)

        for criterion in self.criteria:
            for model in self.models:
                dir = os.path.join(self.dir_processed, f"full_scores/{criterion}/")
                
                os.makedirs(dir, exist_ok=True)

                dir = os.path.join(self.dir_processed, f"{criterion}/")
                os.makedirs(dir, exist_ok=True)

    def run(self):
        for criterion in self.criteria:
            self.criterion = criterion
            for model in self.models:
                self.model = model
                self.process()

        self.combine_scores()

    def process(self):
        # General post-processing
        self.process_scores()
        
        # Criteria-specific post-processing
        if self.criterion == "fidelity":
            self.process_fidelity()
        elif self.criterion == "temporal":
            self.process_temporal()
        elif self.criterion == "diversity":
            self.process_diversity()
        elif self.criterion == "utility":
            self.process_utility()
        elif self.criterion == "privacy":
            self.process_privacy()

    def process_scores(self):
        print(f"Processing scores..")
        dir = os.path.join(self.dir, f"{self.criterion}/{self.model}/scores/")

        # Combine and save scores of model
        scores = self.files_to_dataframe(dir)
        path = os.path.join(self.dir_processed, f"full_scores/{self.criterion}/{self.model}")
        save_df_to_csv(scores, path)

    def process_fidelity(self): 
        print(f"Processing {self.criterion}..")
        stats_dir = os.path.join(self.dir, f"{self.criterion}/{self.model}/stats_per_var/")
        corrs_dir = os.path.join(self.dir, f"{self.criterion}/{self.model}/correlations/")
        
        # Combine and save stats of model
        df = self.process_dataframe(stats_dir, "fidelity_full_stats")

        # Compute and save average scores of model
        # path = os.path.join(self.processed_dir, f"{self.criterion}/{self.model}_fidelity_avg_stats")
        # avg_stats = df.mean(axis=1)
        # self.save_df_all_formats(avg_stats, path)

        # Combine and save avg correlation matrix of model
        self.process_matrix(corrs_dir, "fidelity_avg_matrix", "Average Variable Correlations")

    def process_temporal(self):
        print(f"Processing {self.criterion}..")
        dir_event_dist = os.path.join(self.dir, f"{self.criterion}/{self.model}/sim_event_distributions/")
        dir_autocorr = os.path.join(self.dir, f"{self.criterion}/{self.model}/sim_autocorrelations/")
        dir_temp_dist = os.path.join(self.dir, f"{self.criterion}/{self.model}/sim_temporal_distances/")

        # Combine and save stats of model
        self.process_dataframe(dir_event_dist, "full_results_sim_event_distributions")

        # Combine and save avg autocorrelation matrix of model
        self.process_matrix(dir_autocorr, "temporal_avg_temporal_distance_matrix", "Average Temporal Correlations")

        # Combine and save avg temporal distance matrix of model
        self.process_matrix(dir_temp_dist, "temporal_avg_temporal_correlation_matrix", "Average Temporal Distances")

    def process_diversity(self):
        print(f"Processing {self.criterion}..")
        pass
    
    def process_utility(self): 
        print(f"Processing {self.criterion}..")
        dir_reg = os.path.join(self.dir, f"{self.criterion}/{self.model}/regression/")
        dir_class_results = os.path.join(self.dir, f"{self.criterion}/{self.model}/classification/results/")
        dir_class_conf_real = os.path.join(self.dir, f"{self.criterion}/{self.model}/classification/conf_matrices/real/")
        dir_class_conf_syn = os.path.join(self.dir, f"{self.criterion}/{self.model}/classification/conf_matrices/syn/")

        # Combine and save full classification and regression results of model
        self.process_dataframe(dir_reg, "full_results_classification")
        self.process_dataframe(dir_class_results, "full_results_regression")

        # Combine and save heatmaps of temporal correlations
        self.process_matrix(dir_class_conf_real, "avg_class_conf_matrix_real", "Average confusion matrix on real data")
        self.process_matrix(dir_class_conf_syn, "avg_class_conf_matrix_syn", "Average confusion matrix on synthetic data")
    
    def process_privacy(self):
        print(f"Processing {self.criterion}..")
        pass

    def combine_scores(self):
        for criterion in self.criteria:
            for model in self.models:
                dir = os.path.join(self.dir_processed, f"full_scores/{criterion}/")
                full_scores = self.files_to_dataframe(dir)

                # Store full scores
                path = os.path.join(self.dir_processed, f"full_scores/full_scores_{criterion}")
                save_df_to_csv(full_scores, path)

                # Store averages
                avg_scores = full_scores.groupby('Model').mean().reset_index()
                df_transposed = avg_scores.set_index('Model').T.round(3)
                path = os.path.join(self.dir_processed, f"avgs_scores/avgs_scores_{criterion}")
                self.save_df_all_formats(df_transposed, path, index=True)

        # Concatenate criteria scores and save
        dir = os.path.join(self.dir_processed, f"full_scores/")
        full_criteria_scores = self.concatenate_criteria(dir)
        df_transposed = full_criteria_scores.set_index('Model').T.round(3)
        path = os.path.join(self.dir_processed, f"full_scores/full_scores")
        self.save_df_all_formats(df_transposed, path, index=True)

        # Compute average scores and save
        avg_scores = full_criteria_scores.groupby('Model').mean().reset_index()
        df_transposed = avg_scores.set_index('Model').T.round(3)
        path = os.path.join(self.dir_processed, f"full_scores/avg_scores")
        self.save_df_all_formats(df_transposed, path, index=True)

    def concatenate_criteria(self, dir):
        scores = pd.DataFrame()
        csv_files = get_csv_files(dir)

        for file in csv_files:
            csv_path = os.path.join(dir, file)
            df = pd.read_csv(csv_path)
            if "Model" in scores.columns:
                df.drop(columns=["Model"], inplace=True)
            scores = pd.concat([scores, df], axis=1)
        return scores

    def files_to_dataframe(self, dir):
        scores = pd.DataFrame()
        csv_files = get_csv_files(dir)

        for file in csv_files:
            csv_path = os.path.join(dir, file)
            df = pd.read_csv(csv_path)
            scores = pd.concat([scores, df], axis=0)
        return scores

    def files_to_matrix(self, dir):
        npy_files = get_npy_files(dir)
        summed_matrix = None
        count = 0

        for file in npy_files:
            path = os.path.join(dir, file)
            matrix = np.load(path)

            if summed_matrix is None:
                summed_matrix = matrix
            else:
                summed_matrix += matrix
            
            count += 1

        average_matrix = summed_matrix / count
        return average_matrix

    def process_matrix(self, dir, filename, title):
        # Combine and save matrix
        matrix = self.files_to_matrix(dir)
        path = os.path.join(self.dir_processed, f"{self.criterion}/{self.model}_{filename}")
        self.visualize_matrix(matrix, title, path)
    
    def process_dataframe(self, dir, filename):
        # Combine and save dataframe
        df = self.files_to_dataframe(dir)
        path = os.path.join(self.dir_processed, f"{self.criterion}/{self.model}_{filename}")
        self.save_df_all_formats(df, path)
        return df

    def save_df_all_formats(self, df, path, index=False):
        save_df_to_csv(df, path, index)
        save_df_to_latex(df, path, index)
        save_df_to_markdown(df, path, index)

    def visualize_matrix(self, matrix, title, path):
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.savefig(f"{path}.png")
        plt.close()




