from src.data.data_loader import get_data
from src.models.rgan import RGAN, train_RGAN, RGAN_params

import torch

if __name__ == "__main__":
    train_data, test_data, shape = get_data("cf_classification", 0.7)

    RGAN_params["epochs"] = 200
    model = RGAN(shape, **RGAN_params)

    path = f"outputs/testing/RGAN/"

    train_RGAN(model, train_data, path)

    syndata = model.generate_data(1000, shape[1], shape[2])

    torch.save(syndata, f"outputs/testing/RGAN/syn.pt")


    