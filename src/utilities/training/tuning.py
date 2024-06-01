import optuna
import os
import json

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from src.utilities.training.hyperparameters import get_grid
from src.utilities.utils import create_directory
from src.models.models import train_model

from datetime import datetime

def objective(model_class, trial, dataset: Dataset, path:str, n_folds: int):
    
    # Set hyperparameter grid
    hyperparams = get_grid(model_class, trial)
    hyperparams.update({
        "num_sequences": dataset.sequences.shape[0],
        "num_events": dataset.sequences.shape[1],
        "num_features": dataset.sequences.shape[2],
    })

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Starting trial {trial.number}, fold {fold}")
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        
        model = model_class(**hyperparams)

        moment = datetime.now().strftime('%Y-%m-%d_%H-%M')
        # Set paths and create directories
        fold_log_dir = f"{path}runs/{dataset.NAME}-{model.NAME}/{moment}/trial_{trial.number}_fold_{fold}/"
        loss_log_dir = f"{path}losses/{dataset.NAME}-{model.NAME}/{moment}/trial_{trial.number}_fold_{fold}/"
        create_directory(fold_log_dir)
        create_directory(loss_log_dir)

        # Train classifier
        fold_val_loss = train_model(model, train_data, fold_log_dir, loss_log_dir, val_data)

        val_losses.append(fold_val_loss)

    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss


def optimize_hyperparameters(dataset, model, output_path: str, n_trials=10, n_folds=5):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(model, trial, dataset, output_path, n_folds), n_trials=n_trials)

    return study.best_trial

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

