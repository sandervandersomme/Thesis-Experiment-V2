from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.utils import get_csv_files, save_df_to_csv, save_df_to_latex, save_df_to_markdown

class PostProcessor():
    def __init__(self, criteria, models, dir: str) -> None:
        self.dir = dir
        self.criteria = criteria
        self.models = models

    def run(self):
        for criterion in self.criteria:
            self.criterion = criterion
            for model in self.models:
                self.model = model
                self.process()

        # TODO: Combine all results into one big table

    def process(self):
        # General post-processing
        self.process_scores()
        
        # # Criteria-specific post-processing
        # if self.criterion == "fidelity":
        #     self.process_fidelity(self)
        # elif self.criterion == "temporal":
        #     self.process_temporal(self)
        # elif self.criterion == "diversity":
        #     self.process_diversity(self)
        # elif self.criterion == "utility":
        #     self.process_utility(self)
        # elif self.criterion == "privacy":
        #     self.process_privacy(self)

    def process_scores(self):
        dir = os.path.join(self.dir, f"{self.criterion}/{self.model}/scores/")

        scores = pd.DataFrame()
        csv_files = get_csv_files(dir)

        for file in csv_files:
            csv_path = os.path.join(dir, file)
            df = pd.read_csv(csv_path)
            scores = pd.concat([scores, df], axis=0)

        scores.set_index('Model', inplace=True)
        path = os.path.join(self.dir, f"{self.criterion}/{self.model}/processed/", "full_scores")
        save_df_to_csv(scores, path)
        save_df_to_latex(scores, path)
        save_df_to_markdown(scores, path)

        path = os.path.join(self.dir, f"{self.criterion}/{self.model}/processed/", "avg_scores")
        avg_scores = scores.mean(axis=1)
        
        save_df_to_csv(avg_scores, path)
        save_df_to_latex(avg_scores, path)
        save_df_to_markdown(avg_scores, path)

    def process_fidelity(self): 
        stats_dir = os.path.join(self.dir, f"{self.criterion}/{self.model}/stats_per_var/")
        stats = pd.DataFrame()
        csv_files = get_csv_files(dir)

        corrs_dir = os.path.join(self.dir, f"{self.criterion}/{self.model}/correlations/")





    def process_temporal(self): raise NotImplementedError
    def process_diversity(self): raise NotImplementedError
    def process_utility(self): raise NotImplementedError
    def process_privacy(self): raise NotImplementedError  


