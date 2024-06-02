import argparse

from src.data.data_loader import select_data
from src.training.tuning import optimize_hyperparameters

from src.models.models import select_model
from src.paths import TEST_PATH_HYPERPARAMS, PATH_HYPERPARAMS
from src.utilities.utils import load_best_params_and_score, save_trial

from torch.utils.data import Dataset

"""
Script for tuning models, either to get results or to test tuning of them
"""

def parse_args():
    parser = argparse.ArgumentParser()

    # Test mode
    parser.add_argument('--test', action='store_true', help='Is this run in testing mode?')

    # Tuning setup
    parser.add_argument('--dataset', type=str, help='Dataset to be used', default='cf')
    parser.add_argument('--models', nargs='+', type=str, help='Model types to be used')
    parser.add_argument('--trials', type=int, help='Number of trials', default=10)
    parser.add_argument('--folds', type=int, help='Number of k-fold splits', default=5)

    args = parser.parse_args()
    return args

def tune_model(model, dataset: Dataset, trials: int, folds: int, path: str):
    print(f"Using dataset {dataset.NAME} and model {model.NAME}")

    # Load saved trial
    suffix = f"/{dataset.NAME}-{model.NAME}.json"
    saved_trial = load_best_params_and_score(path + suffix)
    new_trial = optimize_hyperparameters(dataset, model, path, trials, folds)

    save_best_trial(saved_trial, new_trial, path + suffix)

    # hyperparams = load_best_params_and_score()
    # instance = model(**hyperparams)
    # Select train function
    # Train model
    # Save model

def save_best_trial(old_trial, new_trial, path:str):
    # If score is better, than overwrite saved params
    if old_trial is None:
        print("Model has not been tuned before, saving hyperparameter..")
        save_trial(new_trial, path)
    else:
        print(f"Old score: {old_trial["value"]}")
        print(f"New score: {new_trial.value}")

        if new_trial.value < old_trial["value"]:
            save_trial(new_trial, path)

        print(f"Best hyperparameters: {new_trial.params}")

def set_path(test):
    # Set path
    if test: return TEST_PATH_HYPERPARAMS
    else: return PATH_HYPERPARAMS

if __name__ == "__main__":
    # Parse arguments, path, models and dataset
    args = parse_args()
    models = [select_model(model) for model in args.models]
    dataset = select_data(args.dataset)

    # Loop through models
    for model_class in args.models:
        # Retrieve model class
        path = set_path(args.test)
        model = select_model(model_class)

        # Tune model
        tune_model(model, dataset, args.trials, args.folds, path)


    
