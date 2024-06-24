# Import tools for utility evaluation
from src.eval.evaluators.evaluator import Evaluator
from src.eval.utility.methods_utility import run_downstream_task

# Import downstream modeling tasks
from src.training.hyperparameters import load_default_params, load_optimal_params
from src.models.models import task_to_model

import pandas as pd
from typing import List
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

class UtilityEvaluator(Evaluator):
    def __init__(self, eval_args, output_dir: str) -> None:
        super().__init__(eval_args, output_dir)         
        self.eval_dir = os.path.join(self.eval_dir, "utility/")

        self.class_results = []
        self.reg_results = []

    def evaluate(self, files: List[str]):
        return super().evaluate(files)

    def _evaluate_dataset(self, syndata: torch.Tensor):
        train_data = self.eval_args.real_data[self.eval_args.train_data.indices]
        test_data = self.eval_args.real_data[self.eval_args.test_data.indices]

        if "classification" in self.eval_args.tasks:
            # Load hyperparameters
            if self.eval_args.flag_default_params: hyperparams = load_default_params("classifier")
            else: hyperparams = load_optimal_params(self.hyperparams_dir, f"{self.eval_args.dataset}-{task_to_model("classification")}-{self.eval_args.seed}")

            results = run_downstream_task("classification", self.eval_args.dataset, syndata, train_data, test_data, self.eval_args.columns, hyperparams, self.eval_args.epochs, self.eval_args.val_split_size, self.eval_args.seed)
            self.class_results.append(results)

        if "regression" in self.eval_args.tasks:
            # Load hyperparameters
            if self.eval_args.flag_default_params: hyperparams = load_default_params("regressor")
            else: hyperparams = load_optimal_params(self.hyperparams_dir, f"{self.eval_args.dataset}-{task_to_model("regression")}-{self.eval_args.seed}")

            results = run_downstream_task("regression", self.eval_args.dataset, syndata, train_data, test_data, self.eval_args.columns, hyperparams, self.eval_args.epochs, self.eval_args.val_split_size, self.eval_args.seed)
            self.reg_results.append(results)

    def _post_processing(self):
        scores_df = pd.DataFrame()

        if len(self.class_results) > 0:
            results = self._post_process_classification()
            scores_df = pd.concat([scores_df, results], axis=1)

        if len(self.reg_results) > 0:
            results = self._post_process_regression()
            scores_df = pd.concat([scores_df, results], axis=1)

        return scores_df

    def _post_process_classification(self):
        print("Post-processing classification results")

        unzipped_results = zip(*self.class_results)
        all_scores, all_raw_results = map(list, unzipped_results)

        # Process scores
        scores_df = pd.DataFrame()
        scores_df["Diff. Accuracy"] = pd.Series([scores["diff_accuracy"] for scores in all_scores])
        scores_df["Diff. Precision"] = pd.Series([scores["diff_precision"] for scores in all_scores])
        scores_df["Diff. Recall"] = pd.Series([scores["diff_recall"] for scores in all_scores])
        scores_df["Diff. F1 Score"] = pd.Series([scores["diff_f1_score"] for scores in all_scores])
        scores_df["Diff. Diff ROC AUC"] = pd.Series([scores["diff_roc_auc"] for scores in all_scores])

        # Process additional results
        raw_results_df = pd.DataFrame()
        raw_results_df["Accuracy Real"] = pd.Series([raw_results["accuracy_real"] for raw_results in all_raw_results])
        raw_results_df["Accuracy Synthetic"] = pd.Series([raw_results["accuracy_synthetic"] for raw_results in all_raw_results])
        raw_results_df["Precision Real"] = pd.Series([raw_results["precision_real"] for raw_results in all_raw_results])
        raw_results_df["Precision Synthetic"] = pd.Series([raw_results["precision_synthetic"] for raw_results in all_raw_results])
        raw_results_df["Recall Real"] = pd.Series([raw_results["recall_real"] for raw_results in all_raw_results])
        raw_results_df["Recall Synthetic"] = pd.Series([raw_results["recall_synthetic"] for raw_results in all_raw_results])
        raw_results_df["F1 Score Real"] = pd.Series([raw_results["f1_score_real"] for raw_results in all_raw_results])
        raw_results_df["F1 Score Synthetic"] = pd.Series([raw_results["f1_score_synthetic"] for raw_results in all_raw_results])
        raw_results_df["ROC AUC Real"] = pd.Series([raw_results["roc_auc_real"] for raw_results in all_raw_results])
        raw_results_df["ROC AUC Synthetic"] = pd.Series([raw_results["roc_auc_synthetic"] for raw_results in all_raw_results])
        save_raw_results_table(raw_results_df, self.eval_dir, f"{self.eval_args.model_type}_raw_classification_scores")

        # Process confusion matrices
        avg_real_confusion_matrices = sum([raw_results["confusion_matrix_real"] for raw_results in all_raw_results]) / len(all_raw_results)
        avg_synthetic_confusion_matrices = sum([raw_results["confusion_matrix_synthetic"] for raw_results in all_raw_results]) / len(all_raw_results)
        avg_diff_matrix = sum([scores["diff_matrix"] for scores in all_scores])  / len(all_scores)

        # Visualize confusion matrices
        self._visualise_confusion_matrix(avg_real_confusion_matrices, "Average Real Confusion Matrix", f"{self.eval_args.model_type}_avg_real_conf_matrix_classification")
        self._visualise_confusion_matrix(avg_synthetic_confusion_matrices, "Average Synthetic Confusion Matrix", f"{self.eval_args.model_type}_avg_syn_conf_matrix_classification")
        self._visualise_confusion_matrix(avg_diff_matrix, "Average Difference Matrix", f"{self.eval_args.model_type}_avg_diff_matrix_classification")

        return scores_df

    def _post_process_regression(self):
        print("Post-processing regression results")
        unzipped_results = zip(*self.reg_results)
        all_scores, all_raw_results = map(list, unzipped_results)

        # Process scores
        scores_df = pd.DataFrame()
        scores_df["Diff. MAE"] = pd.Series([scores["diff_MAE"] for scores in all_scores])
        scores_df["Diff. MSE"] = pd.Series([scores["diff_MSE"] for scores in all_scores])
        scores_df["Diff. RMSE"] = pd.Series([scores["diff_RMSE"] for scores in all_scores])
        scores_df["Diff. R2"] = pd.Series([scores["diff_R2"] for scores in all_scores])

        # Process additional results
        raw_results_df = pd.DataFrame()
        raw_results_df["MAE Real"] = pd.Series([raw_results["MAE_Real"] for raw_results in all_raw_results])
        raw_results_df["MAE Synthetic"] = pd.Series([raw_results["MAE_Synthetic"] for raw_results in all_raw_results])
        raw_results_df["MSE Real"] = pd.Series([raw_results["MSE_Real"] for raw_results in all_raw_results])
        raw_results_df["MSE Synthetic"] = pd.Series([raw_results["MSE_Synthetic"] for raw_results in all_raw_results])
        raw_results_df["RMSE Real"] = pd.Series([raw_results["RMSE_Real"] for raw_results in all_raw_results])
        raw_results_df["RMSE Synthetic"] = pd.Series([raw_results["RMSE_Synthetic"] for raw_results in all_raw_results])
        raw_results_df["R2 Real"] = pd.Series([raw_results["R2_Real"] for raw_results in all_raw_results])
        raw_results_df["R2 Synthetic"] = pd.Series([raw_results["R2_Synthetic"] for raw_results in all_raw_results])
        save_raw_results_table(raw_results_df, self.eval_dir, f"{self.eval_args.model_type}_raw_regression_scores")

        return scores_df

    def _visualise_confusion_matrix(self, matrix, title, filename):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues")
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        path = os.path.join(self.eval_dir, f"{filename}.png")
        plt.savefig(path)
        plt.close()

def save_raw_results_table(df, eval_dir, filename):
    latex_file_path = os.path.join(eval_dir, f"{filename}.tex")
    markdown_file_path = os.path.join(eval_dir, f"{filename}.md")

    # Save to LaTeX
    with open(latex_file_path, 'w') as f:
        f.write(df.to_latex(index=False))

    # Save to Markdown
    with open(markdown_file_path, 'w') as f:
        f.write(df.to_markdown(index=False))

