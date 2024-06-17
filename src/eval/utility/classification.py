from src.models.classifier import TimeseriesClassifier
from src.training.hyperparameters import select_hyperparams
from src.models.models import train_classifier
import torch
from torch.utils.data import Dataset

def run_classification(dataset_name: str, train_data: Dataset, val_data: Dataset, test_data: Dataset, epochs: int):
    hyperparams = select_hyperparams(dataset_name, 'classifier', train_data.sequences.shape)
    classifier = TimeseriesClassifier(**hyperparams)
    train_classifier(classifier, train_data, val_data)

if __name__ == "__main__":
    from src.data.cf import CF
    from src.data.data_processing import split_train_val_test

    data = CF()
    train_data, val_data, test_data = split_train_val_test(data, train_split=0.7, val_split=0.15)

    run_classification(data.NAME, train_data, val_data, test_data)