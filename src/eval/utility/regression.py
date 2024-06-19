from src.models.regressor import TimeseriesRegressor
from src.models.models import train_regressor
from torch.utils.data import Dataset
from src.data.cf_classification import DownstreamDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import numpy as np
from src.data.data_processing import split

def regression_scores(real_train_data: DownstreamDataset, syndata: DownstreamDataset, test_data: DownstreamDataset, epochs: int, hyperparams: dict, val_split_size: float, seed):
    train_data, val_data = split(real_train_data, val_split_size, seed)
    
    predictions_on_real = run_regressor(train_data, val_data, test_data, epochs, hyperparams)
    predictions_on_syn = run_regressor(syndata, val_data, test_data, epochs, hyperparams)
    true_labels = test_data.targets.numpy()

    predictions_on_real = predictions_on_real.numpy()  # Ensure predictions are NumPy arrays
    predictions_on_syn = predictions_on_syn.numpy()

    return {
        'MAE Real': mean_absolute_error(true_labels, predictions_on_real),
        'MAE Synthetic': mean_absolute_error(true_labels, predictions_on_syn),
        'MSE Real': mean_squared_error(true_labels, predictions_on_real),
        'MSE Synthetic': mean_squared_error(true_labels, predictions_on_syn),
        'RMSE Real': np.sqrt(mean_squared_error(true_labels, predictions_on_real)),
        'RMSE Synthetic': np.sqrt(mean_squared_error(true_labels, predictions_on_syn)),
        'R2 Real': r2_score(true_labels, predictions_on_real),
        'R2 Synthetic': r2_score(true_labels, predictions_on_syn),
        'Diff MAE': np.abs(mean_absolute_error(true_labels, predictions_on_real) - mean_absolute_error(true_labels, predictions_on_syn)),
        'Diff MSE': np.abs(mean_squared_error(true_labels, predictions_on_real) - mean_squared_error(true_labels, predictions_on_syn)),
        'Diff RMSE': np.abs(np.sqrt(mean_squared_error(true_labels, predictions_on_real)) - np.sqrt(mean_squared_error(true_labels, predictions_on_syn))),
        'Diff R2': np.abs(r2_score(true_labels, predictions_on_real) - r2_score(true_labels, predictions_on_syn))
    }

def run_regressor(train_data: Dataset, val_data: Dataset, test_data: Dataset, epochs: int, hyperparams: dict):
    # split real data into train and validation

    regressor = TimeseriesRegressor(**hyperparams)
    train_regressor(regressor, train_data, val_data, epochs)

    regressor.eval()
    with torch.no_grad():
        predictions = regressor(test_data.sequences)
        predictions = predictions.squeeze()  # Ensure predictions are of the correct shape

    return predictions
