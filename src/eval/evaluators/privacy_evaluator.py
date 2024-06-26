from src.eval.evaluators.evaluator import Evaluator
from src.utils import save_df_to_csv

# Import privacy methods
from src.eval.privacy.white_box_mia import mia_whitebox_attack
from src.eval.privacy.black_box_mia import mia_blackbox_attack
from src.eval.privacy.reidentification_risk import reidentification_risk
from src.eval.privacy.aia_attack import attribute_disclosure_attack

import os
from typing import List
import torch
import pandas as pd


class PrivacyEvaluator(Evaluator):
    def __init__(self, syndata: torch.Tensor, eval_args, output_dir: str) -> None:
        super().__init__(syndata, eval_args, output_dir)         

    def setup_paths(self):
        # Set root directory
        self.eval_dir = os.path.join(self.eval_dir, f"privacy/{self.eval_args.model_type}")

        # Set filename
        self.filename = f"{self.eval_args.model_id}_{self.eval_args.syndata_id}"

        # Set paths to dirs
        self.path_scores = os.path.join(self.eval_dir, f"scores/")

    def setup_folders(self):
        # Create dirs
        os.makedirs(self.path_scores, exist_ok=True)

    def evaluate(self):
        print(f"Start privacy evaluation of model {self.eval_args.model_type}..")
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]
        test_data = self.eval_args.real_data[self.eval_args.test_data.indices]

        # Create dataframe
        scores = pd.DataFrame()
        scores["Model"] = pd.Series(self.eval_args.model_type)
        scores = pd.concat([scores, self.evaluate_mia_w(train_data, test_data)], axis=1)
        scores = pd.concat([scores, self.evaluate_mia_b(train_data, test_data)], axis=1)
        scores = pd.concat([scores, self.evaluate_aia(train_data)], axis=1)
        # scores = pd.concat([scores, self.evaluate_reid(train_data)], axis=1)

        # # Store metric scores
        path = os.path.join(self.path_scores, self.filename)
        save_df_to_csv(scores, path)

    def evaluate_mia_w(self, train_data, test_data):
        print("Evaluating MIA whitebox attack..")
        results = mia_whitebox_attack(train_data, test_data, self.eval_args.model, self.eval_args.mia_threshold)
        return pd.Series(results)
        
    def evaluate_mia_b(self, train_data, test_data):
        print("Evaluating MIA blackbox attack..")
        results = mia_blackbox_attack(self.syndata, train_data, test_data, self.eval_args.model, self.eval_args.mia_threshold, self.eval_args.epochs)
        return pd.Series(results)
    
    def evaluate_aia(self, train_data):
        print("Evaluating AIA..")
        results = attribute_disclosure_attack(self.syndata, train_data, self.eval_args.n_neighbors_privacy, self.eval_args.aia_threshold, self.eval_args.num_disclosed_attributes)
        return pd.Series(results)
    
    def evaluate_reid(self, train_data):
        print("Evaluating Reidentification risk..")
        results = reidentification_risk(self.syndata, train_data, self.eval_args.dtw_threshold)
        return pd.Series(results)