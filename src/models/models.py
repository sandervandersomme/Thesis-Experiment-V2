from src.models.downstream.regressor import TimeseriesRegressor, train_regressor
from src.models.downstream.classifier import TimeseriesClassifier, train_classifier

from src.models.generative.rgan import RGAN, train_RGAN, RGAN_params
from src.models.generative.rwgan import RWGAN, train_RWGAN, RWGAN_params
from src.models.generative.timegan import TimeGAN, train_TimeGAN, TimeGAN_params

utility_models = {
    "regressor": (TimeseriesRegressor, train_regressor),
    "classifier": (TimeseriesClassifier, train_classifier)
}

gen_models = {
    "RGAN": (RGAN, train_RGAN, RGAN_params),
    "RWGAN": (RWGAN, train_RWGAN, RWGAN_params),
    "TimeGAN": (TimeGAN, train_TimeGAN, TimeGAN_params)
}

if __name__ == "__main__":
    print(gen_models)
    print(utility_models)