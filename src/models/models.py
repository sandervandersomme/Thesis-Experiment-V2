from src.models.downstream.regressor import TimeseriesRegressor, train_regressor
from src.models.downstream.classifier import TimeseriesClassifier, train_classifier

from src.models.generative.rgan import RGAN, train_RGAN
from src.models.generative.rwgan import RWGAN, train_RWGAN
from src.models.generative.timegan import TimeGAN, train_TimeGAN

def train_model(model, train_data, log_dir, val_data=None):
    if isinstance(model, RGAN):
        return train_RGAN(model, train_data, log_dir)
    elif isinstance(model, TimeGAN):
        return train_TimeGAN(model, train_data, log_dir, val_data)
    elif isinstance(model, RWGAN):
        return train_RWGAN(model, train_data, log_dir)
    elif isinstance(model, TimeseriesClassifier):
        return train_classifier(model, train_data, log_dir, val_data)
    elif isinstance(model, TimeseriesRegressor):
        return train_regressor(model, train_data, log_dir, val_data)