import torch

from src.data.cf import CF
from src.data.cf_classification import CF_Classification
from src.data.cf_regression import CF_Regression
from torch.utils.data import Dataset

def load_syn_data(path): return torch.load(path)

def select_data(dataset: str) -> Dataset:
    "select and load dataset"
    if dataset == "cf":
        return CF()

def create_downstream_dataset(dataset, task: str):
    if task == "classification":
        return create_classification_dataset(dataset)
    if task == "regression":
        return create_regression_dataset(dataset)
    
def create_classification_dataset(dataset):
    if dataset.NAME == "cf":
        return CF_Classification(dataset.sequences, dataset.columns.copy())

def create_regression_dataset(dataset):
    if dataset.NAME == "cf":
        return CF_Regression(dataset.sequences, dataset.columns.copy())