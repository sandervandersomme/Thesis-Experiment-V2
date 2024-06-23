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
        self.args = args

        self.results_full = pd.DataFrame()
        self.results_average = pd.DataFrame()

    def collect_results(self):
        for model_type in self.args.models:
            self.args.model_type = model_type
            model_files = get_filenames_of_models(model_type, self.num_instances)

            self.collect_model_results(model_type, model_files)
        # Save results to files
        self.save_results()

    def collect_model_results(self, model: str, model_files: List[str]):
        print(f"Start evaluating model {model}..")
        # Create new evaluators for model
        self.evaluators = create_evaluators(self.criteria, self.args, self.output_dir)

        # Get files of datasets
        filenames_datasets = get_filenames_of_syndatasets(model, self.num_instances, self.num_datasets)
        
        # Use evaluators for evaluation of the datasets
        for evaluator in self.evaluators:
            avgs_scores, full_scores = evaluator.evaluate(filenames_datasets)
            
            # Add average scores to the results_average dataframe
            avgs_scores['Model'] = model  # Add model information as a column
            self.results_average = pd.concat([self.results_average, avgs_scores], axis=0)

            # Add full scores to the results_full dataframe
            full_scores_df = pd.DataFrame(full_scores)
            full_scores_df['Model'] = model
            self.results_full = pd.concat([self.results_full, full_scores_df], axis=0)
        
    def save_results(self):
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the average results
        avg_results_path_csv = os.path.join(self.output_dir, 'average_results.csv')
        avg_results_path_latex = os.path.join(self.output_dir, 'average_results.tex')
        avg_results_path_md = os.path.join(self.output_dir, 'average_results.md')
        self.results_average.to_csv(avg_results_path_csv, index=False)
        self.results_average.to_latex(avg_results_path_latex, index=False)
        self.save_markdown(self.results_average, avg_results_path_md)
        
        # Save the full results
        full_results_path_csv = os.path.join(self.output_dir, 'full_results.csv')
        full_results_path_latex = os.path.join(self.output_dir, 'full_results.tex')
        full_results_path_md = os.path.join(self.output_dir, 'full_results.md')
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
    if criteria == "all":
        return [
            FidelityEvaluator(args, output_dir)
        ]
    else: 
        for criterion in criteria:
            evaluators = []
            if criterion == "fidelity":
                evaluators.append(FidelityEvaluator(args, output_dir))
            if criterion == "temporal":
                evaluators.append(TemporalFidelityEvaluator(args, output_dir))
            if criterion == "diversity":
                raise NotImplementedError
            if criterion == "utility":
                raise NotImplementedError
            if criterion == "privacy":
                raise NotImplementedError
    return evaluators
