from src.eval.evaluators.evaluator import Evaluator

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
    def __init__(self, eval_args, output_dir: str) -> None:
        super().__init__(eval_args, output_dir)         
        self.eval_dir = os.path.join(self.eval_dir, "privacy/")

        self.all_mia_w_results = []
        self.all_mia_b_results = []
        self.all_aia_results = []
        self.all_reid_risk = []

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor):
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]
        test_data = self.eval_args.real_data[self.eval_args.test_data.indices]

        self.all_mia_w_results.append(mia_whitebox_attack(train_data, test_data, self.eval_args.model, self.eval_args.mia_threshold))
        self.all_mia_b_results.append(mia_blackbox_attack(syndata, train_data, test_data, self.eval_args.model, self.eval_args.mia_threshold, self.eval_args.epochs))
        self.all_aia_results.append(attribute_disclosure_attack(syndata, train_data, self.eval_args.n_neighbors_privacy, self.eval_args.aia_threshold, self.eval_args.num_disclosed_attributes))
        self.all_reid_risk.append(reidentification_risk(syndata, train_data, self.eval_args.dtw_threshold))

    def _post_processing(self):
        print("Post-processing classification results")

        # Initialize dataframe
        scores_df = pd.DataFrame()

        # Process MIA Whitebox scores
        scores_df["MIA White TPR"] = pd.Series([results["tpr"] for results in self.all_mia_w_results])
        scores_df["MIA White FPR"] = pd.Series([results["fpr"] for results in self.all_mia_w_results])
        scores_df["MIA White Accuracy"] = pd.Series([results["accuracy"] for results in self.all_mia_w_results])
        scores_df["MIA White Balanced Accuracy Advantage"] = pd.Series([results["balanced_accuracy_advantage"] for results in self.all_mia_w_results])

        # Process MIA Whitebox scores
        scores_df["MIA Black TPR"] = pd.Series([results["tpr"] for results in self.all_mia_b_results])
        scores_df["MIA Black FPR"] = pd.Series([results["fpr"] for results in self.all_mia_b_results])
        scores_df["MIA Black Accuracy"] = pd.Series([results["accuracy"] for results in self.all_mia_b_results])
        scores_df["MIA Black Balanced Accuracy Advantage"] = pd.Series([results["balanced_accuracy_advantage"] for results in self.all_mia_b_results])

        # Process AIA scores
        scores_df["MIA Black TPR"] = pd.Series(self.all_aia_results)

        # Process Reidentification risk scores
        scores_df["Reid. Precision"] = pd.Series([results["precision"] for results in self.all_mia_b_results])
        scores_df["Reid. Recall"] = pd.Series([results["recall"] for results in self.all_mia_b_results])
        scores_df["Reid. MSE"] = pd.Series([results["mse"] for results in self.all_mia_b_results])
        scores_df["Reid. AUC ROC"] = pd.Series([results["auc_roc"] for results in self.all_mia_b_results])

        return scores_df