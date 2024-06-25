from typing import List
import os
from src.utils import get_filenames_of_models, get_filenames_of_syndatasets, load_model

# Import evaluators
from src.eval.evaluators.evaluator import Evaluator
from src.eval.evaluators.fidelity_evaluator import FidelityEvaluator
from src.eval.evaluators.temporal_fidelity_evaluator import TemporalFidelityEvaluator
from src.eval.evaluators.diversity_evaluator import DiversityEvaluator
from src.eval.evaluators.utility_evaluator import UtilityEvaluator
from src.eval.evaluators.privacy_evaluator import PrivacyEvaluator

import pandas as pd

class Collector():
    def __init__(self, criteria: List[str], models: List[str], num_instances: int, num_datasets: int, args, output_dir: str) -> None:
        self.criteria = criteria
        self.models = models
        self.num_instances = num_instances
        self.num_datasets = num_datasets

        self.output_dir = output_dir
        self.eval_dir = os.path.join(output_dir, 'eval/')
        self.model_dir = os.path.join(output_dir, 'models/')
        self.args = args

        self.results_full = pd.DataFrame()
        # self.results_average = pd.DataFrame()

    def collect_results(self):
        for model_type in self.args.models:
            self.args.model_type = model_type
            model_files = get_filenames_of_models(model_type, self.num_instances)

            self.collect_model_results(model_type, model_files)

        self.results_average = self.results_full.groupby('Model').mean().reset_index()
        self.save_results()


    def collect_model_results(self, model: str, model_files: List[str]):
        print(f"Start evaluating model {model}..")

        # Create new evaluators for model
        self.evaluators = create_evaluators(self.criteria, self.args, self.output_dir)

        # Get files of datasets
        filenames_datasets = get_filenames_of_syndatasets(model, self.num_instances, self.num_datasets)
        
        # Initialise dataframe for collecting model results
        model_scores_df = pd.DataFrame()

        # Use evaluators for evaluation of the datasets
        for id in range(len(self.evaluators)):
            evaluator = self.evaluators.pop()
            for filename in model_files:
                self.args.model = load_model(self.model_dir, filename)
                new_full_scores = evaluator.evaluate(filenames_datasets)
                model_scores_df = pd.concat([model_scores_df, new_full_scores], axis=1)

        # Add model information
        model_scores_df['Model'] = model

        # Add model results to all results
        self.results_full = pd.concat([self.results_full, model_scores_df], axis=0)


    def save_results(self):
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the average results
        avg_results_path_csv = os.path.join(self.eval_dir, 'average_results.csv')
        avg_results_path_latex = os.path.join(self.eval_dir, 'average_results.tex')
        avg_results_path_md = os.path.join(self.eval_dir, 'average_results.md')
        self.results_average.to_csv(avg_results_path_csv, index=False)
        self.results_average.to_latex(avg_results_path_latex, index=False)
        self.save_markdown(self.results_average, avg_results_path_md)
        
        # Save the full results
        full_results_path_csv = os.path.join(self.eval_dir, 'full_results.csv')
        full_results_path_latex = os.path.join(self.eval_dir, 'full_results.tex')
        full_results_path_md = os.path.join(self.eval_dir, 'full_results.md')
        self.results_full.to_csv(full_results_path_csv, index=False)
        self.results_full.to_latex(full_results_path_latex, index=False)
        self.save_markdown(self.results_full, full_results_path_md)

    def save_markdown(self, df: pd.DataFrame, path: str):
        with open(path, 'w') as f:
            f.write(self.dataframe_to_markdown(df))

    def dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        markdown = df.to_markdown(index=False)
        return markdown

def create_evaluators(criteria: List[str], args, output_dir) -> List[Evaluator]:
    if "all" in criteria:
        return [
            FidelityEvaluator(args, output_dir),
            TemporalFidelityEvaluator(args, output_dir),
            DiversityEvaluator(args, output_dir),
            UtilityEvaluator(args, output_dir),
            PrivacyEvaluator(args, output_dir)
        ]
    else: 
        evaluators = []
        for criterion in criteria:
            if criterion == "fidelity":
                evaluators.append(FidelityEvaluator(args, output_dir))
            if criterion == "temporal":
                evaluators.append(TemporalFidelityEvaluator(args, output_dir))
            if criterion == "diversity":
                evaluators.append(DiversityEvaluator(args, output_dir))
            if criterion == "utility":
                evaluators.append(UtilityEvaluator(args, output_dir))
            if criterion == "privacy":
                evaluators.append(PrivacyEvaluator(args, output_dir))
    return evaluators
