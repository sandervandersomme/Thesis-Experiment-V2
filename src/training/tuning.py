import optuna
from optuna.pruners import MedianPruner

import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from src.training.hyperparameters import get_gen_grid, get_downstream_grid
from src.models.models import train_gen_model, train_downstream_model, load_gen_model, load_downstream_model

class Tuner:
    def __init__(self, dataset: Dataset, name_data: str, seed: int, dir: str):
        self.name_data = name_data
        self.dataset = dataset
        self.seed = seed
        self.parameter_folder = dir

    def tune(self):
        raise NotImplementedError

class GenTuner(Tuner):
    def __init__(self, dataset: Dataset, name_data: str, seed: int, dir: str):
        super().__init__(dataset, name_data, seed, dir)

    def objective(self, trial, model_class: str, epochs: int):
        self.shape = (len(self.dataset), *self.dataset[0].shape)
        hyperparams = get_gen_grid(model_class, trial, self.shape)
        val_losses = []

        print(f"Starting trial {trial.number}")

        # Load and train model
        model = load_gen_model(model_class, hyperparams)
        fold_val_loss = train_gen_model(model, self.dataset, epochs, verbose=False)
        val_losses.append(fold_val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss
    
    def tune(self, model: str, trials: int, epochs: int):
        suffix = f"{self.name_data}-{model}-{self.seed}"
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{self.parameter_folder}{suffix}.db')

        study = optuna.create_study(
            study_name=suffix,  
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=MedianPruner()
        )
        study.optimize(lambda trial: self.objective(trial, model, epochs), n_trials=trials)


class DownstreamTuner(Tuner):
    def __init__(self, dataset: Dataset, name_data: str, seed: int, dir: str):
        super().__init__(dataset, name_data, seed, dir)

    def objective(self, trial, model_class: str, folds: int, epochs: int):
        hyperparams = get_downstream_grid(model_class, trial, self.dataset.sequences.shape)
        val_losses = []

        # Cross-validation
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"Starting trial {trial.number}, fold {fold}")

            # Store data in datasets
            train_data = Subset(self.dataset, train_idx)
            val_data = Subset(self.dataset, val_idx)

            # Load and train model
            model = load_downstream_model(model_class, hyperparams)
            fold_val_loss = train_downstream_model(model, train_data, epochs, val_data, verbose=False)
            val_losses.append(fold_val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss

    def tune(self, model: str, trials: int, folds: int, epochs: int):
        suffix = f"{self.name_data}-{model}-{self.seed}"
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{self.parameter_folder}{suffix}.db')

        study = optuna.create_study(
            study_name=suffix,  
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=MedianPruner()
        )
        study.optimize(lambda trial: self.objective(trial, model, folds, epochs), n_trials=trials)