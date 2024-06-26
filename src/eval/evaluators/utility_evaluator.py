# Import tools for utility evaluation
from src.eval.evaluators.evaluator import Evaluator
from src.eval.utility.methods_utility import run_downstream_task

# Import downstream modeling tasks
from src.training.hyperparameters import load_default_params, load_optimal_params
from src.models.models import task_to_model
from src.utils import save_df_to_csv, save_matrix_to_np

import pandas as pd
import torch
import os

class UtilityEvaluator(Evaluator):
    def __init__(self, syndata: torch.Tensor, eval_args, output_dir: str) -> None:
        super().__init__(syndata, eval_args, output_dir)         

    def setup_paths(self):
        # Set root directory
        self.eval_dir = os.path.join(self.eval_dir, f"utility/{self.eval_args.model_type}")

        # Set filename
        self.filename = f"{self.eval_args.model_id}_{self.eval_args.syndata_id}"

        # Set paths to dirs
        self.path_scores = os.path.join(self.eval_dir, f"scores/")
        self.path_class_results = os.path.join(self.eval_dir, f"classification/results/")
        self.path_class_matrices_real = os.path.join(self.eval_dir, f"classification/conf_matrices/real/")
        self.path_class_matrices_syn = os.path.join(self.eval_dir, f"classification/conf_matrices/syn/")
        self.path_reg = os.path.join(self.eval_dir, f"regression/")

    def setup_folders(self):
        # Create dirs
        os.makedirs(self.path_scores, exist_ok=True)
        os.makedirs(self.path_class_results, exist_ok=True)
        os.makedirs(self.path_class_matrices_real, exist_ok=True)
        os.makedirs(self.path_class_matrices_syn, exist_ok=True)
        os.makedirs(self.path_reg, exist_ok=True)

    def evaluate(self):
        print(f"Start utility evaluation of model {self.eval_args.model_type}..")

        # Create dataframe
        scores = pd.DataFrame()
        scores["Model"] = pd.Series(self.eval_args.model_type)

        if "classification" in self.eval_args.tasks:
            scores = pd.concat([scores, self.evaluate_classification()], axis=1)
        
        if "regression" in self.eval_args.tasks:
            scores = pd.concat([scores, self.evaluate_regression()], axis=1)
        
        # # Store metric scores
        path = os.path.join(self.path_scores, self.filename)
        save_df_to_csv(scores, path)

    def evaluate_classification(self):
        print(f"Evaluating classification..")
        
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]
        test_data = self.eval_args.real_data[self.eval_args.test_data.indices]

        # Load hyperparameters
        if self.eval_args.flag_default_params: hyperparams = load_default_params("classifier")
        else: hyperparams = load_optimal_params(self.hyperparams_dir, f"{self.eval_args.dataset}-{task_to_model("classification")}-{self.eval_args.seed}")

        # Perform classification
        scores, raw_results = run_downstream_task("classification", self.eval_args.dataset, self.syndata, train_data, test_data, self.eval_args.columns, hyperparams, self.eval_args.epochs, self.eval_args.val_split_size, self.eval_args.seed)

        # Process additional results
        raw_results_df = pd.DataFrame()
        raw_results_df["Accuracy Real"] = pd.Series(raw_results["accuracy_real"])
        raw_results_df["Accuracy Synthetic"] = pd.Series(raw_results["accuracy_synthetic"])
        raw_results_df["Precision Real"] = pd.Series(raw_results["precision_real"])
        raw_results_df["Precision Synthetic"] = pd.Series(raw_results["precision_synthetic"])
        raw_results_df["Recall Real"] = pd.Series(raw_results["recall_real"])
        raw_results_df["Recall Synthetic"] = pd.Series(raw_results["recall_synthetic"])
        raw_results_df["F1 Score Real"] = pd.Series(raw_results["f1_score_real"])
        raw_results_df["F1 Score Synthetic"] = pd.Series(raw_results["f1_score_synthetic"])
        raw_results_df["ROC AUC Real"] = pd.Series(raw_results["roc_auc_real"])
        raw_results_df["ROC AUC Synthetic"] = pd.Series(raw_results["roc_auc_synthetic"])
        path = os.path.join(self.path_class_results, self.filename)
        save_df_to_csv(raw_results_df, path)
        del raw_results_df

        # Store matrices
        path = os.path.join(self.path_class_matrices_real, self.filename)
        save_matrix_to_np(raw_results["confusion_matrix_real"], path)
        path = os.path.join(self.path_class_matrices_syn, self.filename)
        save_matrix_to_np(raw_results["confusion_matrix_synthetic"], path)

        # Process scores
        scores_df = pd.DataFrame()
        scores_df["Diff. Accuracy"] = pd.Series(scores["diff_accuracy"])
        scores_df["Diff. Precision"] = pd.Series(scores["diff_precision"])
        scores_df["Diff. Recall"] = pd.Series(scores["diff_recall"])
        scores_df["Diff. F1 Score"] = pd.Series(scores["diff_f1_score"])
        scores_df["Diff. Diff ROC AUC"] = pd.Series(scores["diff_roc_auc"])

        return scores_df

    def evaluate_regression(self):
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]
        test_data = self.eval_args.real_data[self.eval_args.test_data.indices]

        # Load hyperparameters
        if self.eval_args.flag_default_params: hyperparams = load_default_params("regressor")
        else: hyperparams = load_optimal_params(self.hyperparams_dir, f"{self.eval_args.dataset}-{task_to_model("regression")}-{self.eval_args.seed}")

        scores, raw_results = run_downstream_task("regression", self.eval_args.dataset, self.syndata, train_data, test_data, self.eval_args.columns, hyperparams, self.eval_args.epochs, self.eval_args.val_split_size, self.eval_args.seed)
        
        # Process additional results
        raw_results_df = pd.DataFrame()
        raw_results_df["MAE Real"] = pd.Series(raw_results["MAE_Real"])
        raw_results_df["MAE Synthetic"] = pd.Series(raw_results["MAE_Synthetic"])
        raw_results_df["MSE Real"] = pd.Series(raw_results["MSE_Real"])
        raw_results_df["MSE Synthetic"] = pd.Series(raw_results["MSE_Synthetic"])
        raw_results_df["RMSE Real"] = pd.Series(raw_results["RMSE_Real"])
        raw_results_df["RMSE Synthetic"] = pd.Series(raw_results["RMSE_Synthetic"])
        raw_results_df["R2 Real"] = pd.Series(raw_results["R2_Real"])
        raw_results_df["R2 Synthetic"] = pd.Series(raw_results["R2_Synthetic"])
        path = os.path.join(self.path_reg, self.filename)
        save_df_to_csv(raw_results_df, path)
        del raw_results_df

        # Process scores
        scores_df = pd.DataFrame()
        scores_df["Diff. MAE"] = pd.Series(scores["diff_MAE"])
        scores_df["Diff. MSE"] = pd.Series(scores["diff_MSE"])
        scores_df["Diff. RMSE"] = pd.Series(scores["diff_RMSE"])
        scores_df["Diff. R2"] = pd.Series(scores["diff_R2"])

        return scores_df