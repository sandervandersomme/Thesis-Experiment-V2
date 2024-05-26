import torch
from typing import Callable

# Load models
from src.models.gen_model import GenModel
from src.models.rgan import RGAN, train_RGAN, RGAN_params
from src.models.rwgan import RWGAN, train_RWGAN, RWGAN_params
from src.models.timegan import TimeGAN, train_TimeGAN, TimeGAN_params

# Load datasets
from torch.utils.data import Dataset
from src.data.cf import CF
from src.data.cf_classification import CF_Classification

RGANpath = f"outputs/testing/RGAN/_"
RWGANpath = f"outputs/testing/RWGAN/_"
TimeGANpath = f"outputs/testing/TimeGAN/"

EPOCHS = 10

def train_and_generate(dataset: Dataset, model_class: GenModel, train_func: Callable, path: str, **params: dict):
    model = model_class(dataset.sequences.shape, **params)
    train_func(model, dataset, path)
    syndata = model.generate_data(1000)

    torch.save(syndata, f"outputs/testing/{model.__MODEL__}/syn.pt")

if __name__ == "__main__":
    
    cf = CF()

    RGAN_params["epochs"] = 10
    RWGAN_params["epochs"] = 10
    TimeGAN_params["epochs"] = 10
# 
    # train_and_generate(cf, RGAN, train_RGAN, RGANpath, **RGAN_params)
    train_and_generate(cf, RWGAN, train_RWGAN, RWGANpath, **RWGAN_params)
    train_and_generate(cf, TimeGAN, train_TimeGAN, TimeGANpath, **TimeGAN_params)

    
