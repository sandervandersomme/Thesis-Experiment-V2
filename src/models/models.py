import torch

from src.models.regressor import TimeseriesRegressor, train_regressor
from src.models.classifier import TimeseriesClassifier, train_classifier

from src.models.rgan import RGAN, train_RGAN
from src.models.rwgan import RWGAN, train_RWGAN
from src.models.timegan import TimeGAN, train_TimeGAN

from src.models.gen_model import GenModel
from src.models.downsteam_model import DownstreamModel

def train_model(model, train_data: torch.Tensor, epochs:int, val_data: torch.Tensor=None, log_run_dir: str=None, log_loss_dir: str=None):
    if isinstance(model, RGAN):
        return train_RGAN(model, train_data, epochs, log_run_dir, log_loss_dir)
    elif isinstance(model, TimeGAN):
        return train_TimeGAN(model, train_data, epochs, val_data, log_run_dir, log_loss_dir)
    elif isinstance(model, RWGAN):
        return train_RWGAN(model, train_data, epochs, log_run_dir, log_loss_dir)
    elif isinstance(model, TimeseriesClassifier):
        return train_classifier(model, train_data, val_data, epochs, log_run_dir, log_loss_dir)
    elif isinstance(model, TimeseriesRegressor):
        return train_regressor(model, train_data, val_data, epochs, log_run_dir, log_loss_dir)
    
downstream_models = {
    "classifier": TimeseriesClassifier,
    "regressor": TimeseriesRegressor
}

gen_models = {
    "rgan": RGAN,
    "rwgan": RWGAN,
    "timegan": TimeGAN
}

def select_downstream_model(model: str) -> DownstreamModel:
    return downstream_models[model]

def select_gen_model(model: str) -> GenModel:
    return gen_models[model]