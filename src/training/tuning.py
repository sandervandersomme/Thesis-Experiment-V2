import optuna
from optuna.pruners import MedianPruner

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

from src.training.hyperparameters import get_grid
from src.models.models import train_gen_model, train_downstream_model

class Tuner:
    def __init__(self, dataset: Dataset, name_data: str, seed: int, exp_dir: str):
        self.name_data = name_data
        self.dataset = dataset
        self.seed = seed
        self.parameter_folder = exp_dir + "hyperparams/"

    def tune(self, model, trials, folds, epochs):
        suffix = f"{self.name_data}-{model.NAME}-{self.seed}"
        print(f"sqlite:///{self.parameter_folder}trials/{suffix}.db")
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{self.parameter_folder}trials/{suffix}.db')

        study = optuna.create_study(
            study_name=suffix,  
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=MedianPruner()
        )
        study.optimize(lambda trial: self.objective(trial, model, folds, epochs), n_trials=trials)

class GenTuner(Tuner):
    def __init__(self, dataset: Dataset, name_data: str, seed: int, exp_dir: str):
        super().__init__(dataset, name_data, seed, exp_dir)

    def objective(self, trial, model_class, folds, epochs):
        self.shape = (len(self.dataset), *self.dataset[0].shape)
        hyperparams = get_grid(model_class, trial, self.shape)
        val_losses = []

        # Cross-validation
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        for fold, (train_idx, _) in enumerate(kfold.split(self.dataset)):
            print(f"Starting trial {trial.number}, fold {fold}")

            # Store data in datasets
            train_data = Subset(self.dataset, train_idx)

            # Load and train model
            model = model_class(**hyperparams)
            fold_val_loss = train_gen_model(model, train_data, epochs)
            val_losses.append(fold_val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss

class DownstreamTuner(Tuner):
    def __init__(self, dataset: Dataset, name_data: str, seed: int, exp_dir: str):
        super().__init__(dataset, name_data, seed, exp_dir)

    def objective(self, trial, model_class, folds, epochs):
        hyperparams = get_grid(model_class, trial, self.dataset.sequences.shape)
        val_losses = []

        # Cross-validation
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"Starting trial {trial.number}, fold {fold}")

            # Store data in datasets
            train_data = Subset(self.dataset, train_idx)
            val_data = Subset(self.dataset, val_idx)

            # Load and train model
            model = model_class(**hyperparams)
            fold_val_loss = train_downstream_model(model, train_data, epochs, val_data)
            val_losses.append(fold_val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss
