from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

from src.utils import get_csv_files, save_df_to_csv, save_df_to_latex, save_df_to_markdown, get_npy_files

class PostProcessor():
    def __init__(self, criteria, models, root_dir: str) -> None:
        self.criteria = criteria
        self.models = models

        # Setup paths and folders
        self.root_dir = root_dir
        self.dir_eval = os.path.join(self.root_dir, f"eval/")
        self.dir_results = os.path.join(self.root_dir, f"results/")
        self.dir_avg_results = os.path.join(self.dir_results, f"avgs/")
        self.dir_full_results = os.path.join(self.dir_results, f"full/")
        self.setup_folders()

    def setup_folders(self):
        # Reset folder
        shutil.rmtree(self.dir_results)
        os.makedirs(self.dir_results)
        os.makedirs(self.dir_avg_results)
        os.makedirs(self.dir_full_results)
    

        # Create folder for each criteria to store intermediate files
        for criterion in self.criteria:
            for model in self.models:
                dir = os.path.join(self.dir_results, f"{criterion}/")
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
        print(f"Processing {self.criterion}..")
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
        dir = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/scores/")

        # Combine and save scores of model
        scores = self.files_to_dataframe(dir)
        path = os.path.join(self.dir_results, f"{self.criterion}/{self.model}")
        save_df_to_csv(scores, path)

    def process_fidelity(self): 
        # Input paths
        stats_dir = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/stats_per_var/")
        corrs_dir = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/correlations/")
        
        # Output paths
        stats_output_path = os.path.join(self.dir_results, f"{self.criterion}/stats/")
        corrs_output_path = os.path.join(self.dir_results, f"{self.criterion}/corrs/")
        os.makedirs(stats_output_path, exist_ok=True)
        os.makedirs(corrs_output_path, exist_ok=True)

        df = self.process_dataframe(stats_dir, f"stats/{self.model}")
        self.process_matrix(corrs_dir, f"corrs/{self.model}", "Average Variable Correlations")

    def process_temporal(self):
        # Input paths
        dir_event_dist = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/sim_event_distributions/")
        dir_autocorr = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/sim_autocorrelations/")
        dir_temp_dist = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/sim_temporal_distances/")
        
        # Output paths
        stats_output_path = os.path.join(self.dir_results, f"{self.criterion}/ev_dist/")
        corrs_output_path = os.path.join(self.dir_results, f"{self.criterion}/autocorr/")
        dists_output_path = os.path.join(self.dir_results, f"{self.criterion}/tempdist/")
        os.makedirs(stats_output_path, exist_ok=True)
        os.makedirs(corrs_output_path, exist_ok=True)
        os.makedirs(dists_output_path, exist_ok=True)

        self.process_dataframe(dir_event_dist, f"ev_dist/{self.model}")
        self.process_matrix(dir_autocorr, f"autocorr/{self.model}", "Average Temporal Correlations")
        self.process_matrix(dir_temp_dist, f"tempdist/{self.model}", "Average Temporal Distances")

    def process_diversity(self):
        pass
    
    def process_utility(self): 
        # Input paths
        dir_reg = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/regression/")
        dir_class_results = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/classification/results/")
        dir_class_conf_real = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/classification/conf_matrices/real/")
        dir_class_conf_syn = os.path.join(self.dir_eval, f"{self.criterion}/{self.model}/classification/conf_matrices/syn/")

        # Output paths
        stats_output_path = os.path.join(self.dir_results, f"{self.criterion}/reg/")
        class_output_path = os.path.join(self.dir_results, f"{self.criterion}/class/")
        confr_output_path = os.path.join(self.dir_results, f"{self.criterion}/class_conf_real")
        confs_output_path = os.path.join(self.dir_results, f"{self.criterion}/class_conf_syn")
        os.makedirs(stats_output_path, exist_ok=True)
        os.makedirs(class_output_path, exist_ok=True)
        os.makedirs(confr_output_path, exist_ok=True)
        os.makedirs(confs_output_path, exist_ok=True)

        # Process data
        self.process_dataframe(dir_reg, f"reg/{self.model}")
        self.process_dataframe(dir_class_results, f"class/{self.model}")
        self.process_matrix(dir_class_conf_real, f"class_conf_real/{self.model}", "Average confusion matrix on real data")
        self.process_matrix(dir_class_conf_syn, f"class_conf_syn/{self.model}", "Average confusion matrix on synthetic data")
    
    def process_privacy(self):
        pass

    def combine_scores(self):
        for criterion in self.criteria:
            for model in self.models:
                dir = os.path.join(self.dir_results, f"{criterion}/")
                full_scores = self.files_to_dataframe(dir)

                # Store full scores
                path = os.path.join(self.dir_results, f"full/scores_{criterion}")
                save_df_to_csv(full_scores, path)

                # Store averages
                avg_scores = full_scores.groupby('Model').mean().reset_index()
                df_transposed = avg_scores.set_index('Model').T.round(3)
                path = os.path.join(self.dir_results, f"avgs/scores_{criterion}")
                save_df_to_csv(df_transposed, path, index=True)
            
            shutil.rmtree(dir)

        # Concatenate criteria scores and save
        full_criteria_scores = self.concatenate_criteria(self.dir_full_results)
        df_transposed = full_criteria_scores.set_index('Model').T.round(3)
        path = os.path.join(self.dir_full_results, f"all_scores")
        self.save_df_all_formats(df_transposed, path, index=True)

        # Compute average scores and save
        avg_scores = full_criteria_scores.groupby('Model').mean().reset_index()
        df_transposed = avg_scores.set_index('Model').T.round(3)
        path = os.path.join(self.dir_avg_results, f"all_avg_scores")
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
        path = os.path.join(self.dir_results, f"{self.criterion}/{filename}")
        self.visualize_matrix(matrix, title, path)
    
    def process_dataframe(self, dir, filename):
        # Combine and save dataframe
        df = self.files_to_dataframe(dir)
        path = os.path.join(self.dir_results, f"{self.criterion}/{filename}")
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




