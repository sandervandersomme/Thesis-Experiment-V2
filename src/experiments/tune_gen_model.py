import argparse

# Set paths
from src.paths import PATH_HYPERPARAMS

# Import models
from src.models.models import select_gen_model

# Import data
from src.data.data_loader import select_data

# Setup hyperparameter tuning
from src.utilities.utils import load_best_params_and_score, save_trial
from src.training.tuning import optimize_hyperparameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to be used', default='cf')
    parser.add_argument('--model', type=str, help='Model type to be tuned')
    parser.add_argument('--trials', type=int, help='Number of trials', default=10)
    parser.add_argument('--folds', type=int, help='Number of k-fold splits', default=5)
    args = parser.parse_args()

    # Setup experiment
    dataset = select_data(args.dataset)
    model_class = select_gen_model(args.model)

    # loading, tuning params
    path = PATH_HYPERPARAMS
    suffix = f"{dataset.NAME}-{model_class.NAME}.json"
    saved_trial = load_best_params_and_score(PATH_HYPERPARAMS + suffix)
    new_trial = optimize_hyperparameters(dataset, model_class, path, args.trials, args.folds)

    # Saving params
    if saved_trial is None:
        print("Model has not been tuned before, saving hyperparameter..")
        save_trial(new_trial, path + suffix)
    else:
        print(f"Old score: {saved_trial["value"]}")
        print(f"New score: {new_trial.value}")

        if new_trial.value < saved_trial["value"]:
            save_trial(new_trial, path + suffix)

        print(f"Best hyperparameters: {new_trial.params}")


    
