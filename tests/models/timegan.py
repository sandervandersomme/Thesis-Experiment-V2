from src.data.data_loader import get_data
from src.models.TimeGAN import TimeGAN, train_TimeGAN, TimeGAN_params


if __name__ == "__main__":
    train_data, test_data, shape = get_data("cf_classification", 0.7)

    model = TimeGAN(shape, **TimeGAN_params)

    path = f"outputs/testing/TimeGAN"

    train_TimeGAN(model, train_data, path)


    