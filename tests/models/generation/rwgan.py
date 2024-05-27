from src.data.data_loader import get_data
from src.models.generative.rwgan import RWGAN, train_RWGAN, RWGAN_params


if __name__ == "__main__":
    train_data, test_data, shape = get_data("cf_classification", 0.7)

    model = RWGAN(shape, **RWGAN_params)

    path = f"outputs/testing/RWGAN/"

    train_RWGAN(model, train_data, path)


    