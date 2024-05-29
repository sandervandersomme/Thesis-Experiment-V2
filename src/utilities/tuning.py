import optuna

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from src.utilities.hyperparameters import get_grid
from src.models.models import train_model

def objective(model_class, trial, dataset: Dataset, output_path:str, n_splits=5):
    # Set hyperparameter grid
    hyperparams = get_grid(model_class, trial)
    hyperparams.update({
        "num_sequences": dataset.sequences.shape[0],
        "num_events": dataset.sequences.shape[1],
        "num_features": dataset.sequences.shape[2],
    })

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        
        model = model_class(**hyperparams)
        model.output_path = output_path

        # Train classifier and log loss
        fold_log_dir = f"runs/experiment_1/trial_{trial.number}_fold_{fold}"
        fold_val_loss = train_model(model, train_data, fold_log_dir, val_data)

        val_losses.append(fold_val_loss)

    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss

def optimize_hyperparameters(dataset, model, output_path: str, n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(model, trial, dataset, output_path), n_trials=n_trials)
    return study.best_trial
