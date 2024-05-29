from src.utilities.utils import set_device
from src.models.models import RGAN, TimeGAN, RWGAN, TimeseriesClassifier, TimeseriesRegressor

def get_grid(model_class, trial):
    if issubclass(model_class, RGAN):
        return default_grid(trial)
    elif issubclass(model_class, TimeGAN):
        return timegan_grid(trial)
    elif issubclass(model_class, RWGAN):
        return rwgan_grid(trial)
    elif issubclass(model_class, TimeseriesClassifier):
        return default_grid(trial)
    elif issubclass(model_class, TimeseriesRegressor):
        return default_grid(trial)

def default_grid(trial):
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": 100,  # Set a high value for epochs to allow early stopping
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "device": set_device(),
        "patience": trial.suggest_int("patience", 5, 50),
        "min_delta": trial.suggest_float("min_delta", 0.0, 0.4)
    }

def timegan_grid(trial):
    grid = default_grid(trial)
    grid.update({
        "scaling_factor": trial.suggest_int("scaling_factor", 1, 10),
        "gamma_weight": trial.suggest_float("gamma_weight", 0, 1),
        "disc_loss_threshold": trial.suggest_float("disc_loss_threshold", 0.0, 0.3)
    })
    return grid

def rwgan_grid(trial):
    grid = default_grid(trial)
    grid.update({
        "n_critic": trial.suggest_int("n_critic", 1, 5),
        "clip_value": trial.suggest_float("clip_value", 0.0, 0.2)
    })
    return grid

default_params = { 
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 10,
    "hidden_dim": 10,
    "num_layers": 1,
    "latent_dim": 10,
    "patience": 5,
    "min_delta": 0.05,
}

TimeGAN_params = {
    "scaling_factor": 10,
    "gamma_weight": 1,
    "disc_loss_threshold": 0.15
}

RWGAN_params = {
    "n_critic": 1,
    "clip_value": 0.05
}