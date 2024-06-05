import os

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
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": 50,  # Set a high value for epochs to allow early stopping
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "patience": trial.suggest_int("patience", 5, 10),
        "min_delta": trial.suggest_float("min_delta", 0.001, 0.01)
    }

def timegan_grid(trial):
    grid = default_grid(trial)
    grid.update({
        "latent_dim": trial.suggest_categorical("latent_dim", [16, 32, 64]),
        "scaling_factor": trial.suggest_int("scaling_factor", 1, 10),
        "gamma_weight": trial.suggest_float("gamma_weight", 0, 1),
        "disc_loss_threshold": trial.suggest_float("disc_loss_threshold", 0.0, 0.3)
    })
    return grid

def rwgan_grid(trial):
    grid = default_grid(trial)
    grid.update({
        "n_critic": trial.suggest_int("n_critic", 3, 10),
        "clip_value": trial.suggest_float("clip_value", 0.01, 0.1)
    })
    return grid

def get_default_params(model: str):
    params = default_params
    if model == 'timegan':
        params.update(TimeGAN_params)
    elif model == 'rwgan':
        params.update(RWGAN_params)
    return params

def select_hyperparams(model, tuned_params:bool=True):
    """
    This function returns the hyperparams for the given model.
    If optimal parameters exist, those are returned.
    If they don't exist, but tuned_params is True, the model is tuned first.
    If they don't exist, and tuned_params if False, the model returns default parameters.
    """
    
    # TODO: Check if tuned params exist
    pass

    # TODO: If they don't exist, if tuned_params is true, tune the model
    # TODO: If they don't exist, if tuned_params is false, return default parameters


default_params = { 
    "batch_size": 5,
    "learning_rate": 0.0001,
    "epochs": 50,
    "hidden_dim": 32,
    "num_layers": 1,
    "latent_dim": 10,
    "patience": 5,
    "min_delta": 0.05,
}

TimeGAN_params = {
    "latent_dim": 32,
    "scaling_factor": 10,
    "gamma_weight": 1,
    "disc_loss_threshold": 0.15
}

RWGAN_params = {
    "n_critic": 1,
    "clip_value": 0.05
}
    
def add_shape_to_params(hyperparams: dict, shape: tuple):
    hyperparams.update({
        "num_sequences": shape[0],
        "num_events": shape[1],
        "num_features": shape[2],
    })
    return hyperparams
