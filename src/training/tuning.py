import optuna
from optuna.pruners import MedianPruner

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from src.training.hyperparameters import get_grid, add_shape_to_params
from src.utilities.utils import create_directory, save_trial
from src.models.models import train_model

from datetime import datetime

def objective(model_class, trial, dataset: Dataset, path:str, n_folds: int, epochs: int, moment):
    
    # Set hyperparameter grid
    hyperparams = get_grid(model_class, trial, dataset.sequences.shape)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Starting trial {trial.number}, fold {fold}")
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        
        model = model_class(**hyperparams)

        # Set paths and create directories
        fold_log_dir = f"{path}/runs/{dataset.NAME}-{model.NAME}/{moment}/trial_{trial.number}_fold_{fold}/"
        loss_log_dir = f"{path}/losses/{dataset.NAME}-{model.NAME}/{moment}/trial_{trial.number}_fold_{fold}/"
        create_directory(fold_log_dir)
        create_directory(loss_log_dir)

        # Train classifier
        fold_val_loss = train_model(model, train_data, epochs, val_data, fold_log_dir, loss_log_dir)

        val_losses.append(fold_val_loss)


    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss


def optimize_hyperparameters(dataset, model, output_path: str, epochs: int, n_trials=10, n_folds=5):
    moment = datetime.now().strftime('%Y-%m-%d_%H-%M')
    storage = optuna.storages.RDBStorage(url=f'sqlite:///outputs/hyperparams/trials/{dataset.NAME}-{model.NAME}.db')

    study = optuna.create_study(study_name=f"{dataset.NAME}-{model.NAME}", direction="minimize", storage=storage, load_if_exists=True, pruner=MedianPruner)
    study.optimize(lambda trial: objective(model, trial, dataset, output_path, n_folds, epochs, moment), n_trials=n_trials)

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

