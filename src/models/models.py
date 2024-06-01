from src.models.regressor import TimeseriesRegressor, train_regressor
from src.models.classifier import TimeseriesClassifier, train_classifier

from src.models.rgan import RGAN, train_RGAN
from src.models.rwgan import RWGAN, train_RWGAN
from src.models.timegan import TimeGAN, train_TimeGAN

def train_model(model, train_data, log_run_dir: str, log_loss_dir: str, val_data=None):
    if isinstance(model, RGAN):
        return train_RGAN(model, train_data, log_run_dir, log_loss_dir)
    elif isinstance(model, TimeGAN):
        return train_TimeGAN(model, train_data, log_run_dir, log_loss_dir, val_data)
    elif isinstance(model, RWGAN):
        return train_RWGAN(model, train_data, log_run_dir, log_loss_dir)
    elif isinstance(model, TimeseriesClassifier):
        return train_classifier(model, train_data, log_run_dir, log_loss_dir, val_data)
    elif isinstance(model, TimeseriesRegressor):
        return train_regressor(model, train_data, log_run_dir, log_loss_dir, val_data)
    
models = {
    "rgan": RGAN,
    "rwgan": RWGAN,
    "timegan": TimeGAN,
    "classifier": TimeseriesClassifier,
    "regressor": TimeseriesRegressor
}

def select_model(model: str):
    return models[model]