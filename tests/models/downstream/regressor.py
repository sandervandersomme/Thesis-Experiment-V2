from src.models.regressor import TimeseriesRegressor, train_regressor
from src.data.cf_regression import CF_Regression
from src.models.downsteam_model import downstream_model_params
from src.data.cf import CF
import torch

if __name__ == "__main__":
    # Train model on real
    cf = CF()
    cf_class = CF_Regression(cf.sequences, cf.columns.copy())
    model = TimeseriesRegressor(cf_class.sequences.shape, **downstream_model_params)
    train_regressor(model, cf_class, f"outputs/testing/classifier/2")

    # Train model on synthetic
    syndata = torch.load(f"outputs/testing/RGAN/syn.pt")
    cf_syn = CF_Regression(syndata, cf.columns)
    model2 = TimeseriesRegressor(cf_syn.sequences.shape, **downstream_model_params)
    train_regressor(model2, cf_syn, f"outputs/testing/classifier/2"
)

    


    