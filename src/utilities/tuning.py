import optuna
from typing import Callable

from torch.utils.data import Dataset, Subset

from sklearn.model_selection import KFold

from src.utilities.utils import set_device


def objective(model_class, train_func: Callable, trial, dataset: Dataset, n_splits=5):
    
    hyperparams = {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": 100,  # Set a high value for epochs to allow early stopping
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "device": set_device(),
        "num_sequences": dataset.sequences.shape[0],
        "num_events": dataset.sequences.shape[1],
        "num_features": dataset.sequences.shape[2],
        "patience": 5,
        "min_delta": 0.01
    }

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        model = model_class(**hyperparams)

        # Train classifier and log loss
        fold_log_dir = f"runs/experiment_1/trial_{trial.number}_fold_{fold}"
        fold_val_loss = train_func(model, train_subset, val_subset, log_dir=fold_log_dir)

        val_losses.append(fold_val_loss)

    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss


def optimize_hyperparameters(dataset, model, train_func, n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(model, train_func, trial, dataset), n_trials=n_trials)
    return study.best_trial



