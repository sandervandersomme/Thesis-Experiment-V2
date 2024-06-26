import torch

from src.models.regressor import TimeseriesRegressor, train_regressor
from src.models.classifier import TimeseriesClassifier, train_classifier

from src.models.rgan import RGAN, train_RGAN
from src.models.rwgan import RWGAN, train_RWGAN
from src.models.timegan import TimeGAN, train_TimeGAN

from src.models.gen_model import GenModel
from src.models.downsteam_model import DownstreamModel

def train_gen_model(model, train_data: torch.Tensor, epochs:int, log_loss_dir: str=None, verbose=True):
    if isinstance(model, RGAN):
        return train_RGAN(model, train_data, epochs, log_loss_dir, verbose)
    elif isinstance(model, TimeGAN):
        return train_TimeGAN(model, train_data, epochs, log_loss_dir, verbose)
    elif isinstance(model, RWGAN):
        return train_RWGAN(model, train_data, epochs, log_loss_dir, verbose)

    raise NotImplementedError
    
def train_downstream_model(model, train_data: torch.Tensor, epochs:int, val_data: torch.Tensor=None, log_loss_dir: str=None, verbose=True):
    if isinstance(model, TimeseriesClassifier):
        return train_classifier(model, train_data, val_data, epochs, log_loss_dir, verbose)
    elif isinstance(model, TimeseriesRegressor):
        return train_regressor(model, train_data, val_data, epochs, log_loss_dir, verbose) 

    raise NotImplementedError
    
downstream_models = {
    "classifier": TimeseriesClassifier,
    "regressor": TimeseriesRegressor
}

gen_models = {
    "rgan": RGAN,
    "rwgan": RWGAN,
    "timegan": TimeGAN
}


def load_downstream_model(model: str, hyperparams) -> DownstreamModel:
    if model == "classifier": return TimeseriesClassifier(**hyperparams)
    elif model == "regressor": return TimeseriesRegressor(**hyperparams)

    raise NotImplementedError()

def load_gen_model(model: str, hyperparams) -> GenModel:
    if model == "rgan": return RGAN(**hyperparams)
    elif model == "rwgan": return RWGAN(**hyperparams)
    elif model == "timegan": return TimeGAN(**hyperparams)

    raise NotImplementedError()

def task_to_model(task: str):
    if task == "classification": return "classifier"
    if task == "regression": return "regressor"

    raise NotImplementedError()