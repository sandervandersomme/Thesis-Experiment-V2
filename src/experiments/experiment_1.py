import argparse

# Set paths
from src.paths import PATH_HYPERPARAMS, PATH_EXP1

# Import models
from src.models.models import RGAN, RWGAN, TimeGAN, TimeseriesClassifier, TimeseriesRegressor

# Import data
from src.data.data_loader import select_data, create_downstream_dataset
from src.data.cf_classification import CF_Classification
from src.data.cf_regression import CF_Regression

# Setup hyperparameter tuning
from src.utilities.utils import load_best_params_and_score
from src.utilities.training.tuning import optimize_hyperparameters, save_best_trial




"""
GOAL OF EXPERIMENT:
- Tune gen models
- Tune downstream models
- run gen models without tuning
- generate data of tuned gen models
- generate data of non-tuned gen models
- evaluate all datasets
- compare performances
- store results
"""

def parse_args():
    parser = argparse.ArgumentParser()

    # Tuning setup
    parser.add_argument('--trials', type=int, help='Number of trials', default=10)
    parser.add_argument('--folds', type=int, help='Number of k-fold splits', default=5)

    args = parser.parse_args()
    return args

def tune_gen_model(model_class, dataset, trials, folds):
    # Load saved trial
    path = PATH_HYPERPARAMS
    suffix = f"{dataset.NAME}-{model_class.NAME}.json"
    saved_trial = load_best_params_and_score(path + suffix)
    new_trial = optimize_hyperparameters(dataset, model_class, path, trials, folds)

    save_best_trial(saved_trial, new_trial, path + suffix)

def tune_downstream_model():
    pass

def train_gen_model():
    pass

def generate_data():
    pass

def evaluate_data():
    pass

def store_results():
    pass

if __name__ == "__main__":
    args = parse_args()

    # Experiment setup
    gen_models = [RGAN, RWGAN, TimeGAN]
    down_models = [TimeseriesRegressor, TimeseriesClassifier]
    datasets = [select_data('cf')]

    # Hyperparameter tuning settings
    folds = args.folds
    trials = args.trials

    # Loop through datasets
    for dataset in datasets:

        # Loop through models
        for model_class in gen_models:
            tune_gen_model(model_class, dataset, trials, folds)

        # tune classification model
        class_dataset = create_downstream_dataset(dataset, "classification")
        tune_downstream_model(TimeseriesClassifier, class_dataset)

        # Tune regression model
        reg_dataset = create_downstream_dataset(dataset, "regression")
        tune_downstream_model(TimeseriesRegressor, reg_dataset)

        for model_class in gen_models:
            pass
            # train_gen_model(model_class, dataset)

        for model_class in gen_models:
            pass
            # Load params
            # Create instance of model with params
            # train model
            # generate data
            # evaluate data



