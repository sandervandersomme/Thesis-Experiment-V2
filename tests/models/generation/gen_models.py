import torch
from typing import Callable

# Load models
from src.models.generative.gen_model import GenModel, gen_params
from src.models.generative.rgan import RGAN, train_RGAN
from src.models.generative.rwgan import RWGAN, train_RWGAN
from src.models.generative.timegan import TimeGAN, train_TimeGAN

# Load datasets
from torch.utils.data import Dataset
from src.data.data_loader import select_data

EPOCHS = 10

def train_and_generate(dataset: Dataset, model_class: GenModel, train_func: Callable, epochs, **params: dict):
    # Update parameters
    gen_params["epochs"] = epochs
    gen_params["min_delta"] = 0.2
    gen_params["patience"] = 50
    gen_params.update(params)
    gen_params.update({
        "num_sequences": dataset.sequences.shape[0],
        "num_events": dataset.sequences.shape[1],
        "num_features": dataset.sequences.shape[2]
    })

    # Configure outputs paths
    model = model_class(**gen_params)
    model.output_path = f"outputs/testing/genmodels/"
    log_dir = f"runs/testing/{model.__NAME__}/"

    # Train model
    train_func(model, dataset, log_dir)
    syndata = model.generate_data(1000)

    torch.save(syndata, f"outputs/testing/genmodels/{model.__NAME__}/syndata")

if __name__ == "__main__":
    
    dataset = select_data("cf")
    epochs = 1000

    # train_and_generate(dataset, RGAN, train_RGAN, epochs)
    train_and_generate(dataset, RWGAN, train_RWGAN, epochs, **RWGAN_params)
    # train_and_generate(dataset, TimeGAN, train_TimeGAN, epochs, **TimeGAN_params)

    
