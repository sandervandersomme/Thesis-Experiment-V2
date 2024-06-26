from src.models.classifier import TimeseriesClassifier
from src.models.models import train_classifier
from torch.utils.data import Dataset
from src.data.cf.cf_classification import DownstreamDataset
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import torch
import numpy as np
from src.data.data_processing import split
from src.utils import set_device

def classification_scores(real_train_data: DownstreamDataset, syndata: DownstreamDataset, test_data: DownstreamDataset, epochs: int, hyperparams: dict, val_split_size: float, seed):
    test_labels, testcounts = torch.unique(test_data.targets, return_counts=True)
    train_labels, traincounts = torch.unique(real_train_data.targets, return_counts=True)
    syn_labels, syncounts = torch.unique(syndata.targets, return_counts=True)

    train_data, val_data = split(real_train_data, val_split_size, seed)
    
    print("Training classification model on real data")
    predictions_on_real = run_classifier(train_data, val_data, test_data, epochs, hyperparams, val_split_size).cpu().numpy()
    print("Training classification model on synthetic data")
    predictions_on_syn = run_classifier(syndata, val_data, test_data, epochs, hyperparams, val_split_size).cpu().numpy()
    true_labels = test_data.targets.cpu().numpy()

    raw_results = {
        # "train_counts": (train_labels, testcounts),
        # "test_counts": (test_labels, testcounts),
        # "syn_counts": (syn_labels, syncounts),
        "accuracy_real": accuracy_score(true_labels, predictions_on_real),
        "accuracy_synthetic": accuracy_score(true_labels, predictions_on_syn),
        "precision_real": precision_score(true_labels, predictions_on_real, zero_division=0),
        "precision_synthetic": precision_score(true_labels, predictions_on_syn, zero_division=0),
        "recall_real": recall_score(true_labels, predictions_on_real),
        "recall_synthetic": recall_score(true_labels, predictions_on_syn),
        "f1_score_real": f1_score(true_labels, predictions_on_real, zero_division=0),
        "f1_score_synthetic": f1_score(true_labels, predictions_on_syn, zero_division=0),
        "roc_auc_real": roc_auc_score(true_labels, predictions_on_real),
        "roc_auc_synthetic": roc_auc_score(true_labels, predictions_on_syn),
        "confusion_matrix_synthetic": confusion_matrix(true_labels, predictions_on_syn),
        "confusion_matrix_real": confusion_matrix(true_labels, predictions_on_real)
    }
    
    scores = {
        "diff_accuracy": accuracy_score(true_labels, predictions_on_syn) - accuracy_score(true_labels, predictions_on_real),
        "diff_precision": precision_score(true_labels, predictions_on_syn, zero_division=0) - precision_score(true_labels, predictions_on_real, zero_division=0),
        "diff_recall": recall_score(true_labels, predictions_on_syn) - recall_score(true_labels, predictions_on_real),
        "diff_f1_score": f1_score(true_labels, predictions_on_syn, zero_division=0) - f1_score(true_labels, predictions_on_real, zero_division=0),
        "diff_roc_auc": roc_auc_score(true_labels, predictions_on_syn) - roc_auc_score(true_labels, predictions_on_real),
        "diff_matrix": confusion_matrix(true_labels, predictions_on_syn) - confusion_matrix(true_labels, predictions_on_real)
    }

    return scores, raw_results

def run_classifier(train_data: Dataset, val_data: Dataset, test_data: Dataset, epochs: int, hyperparams: dict, val_split_size: float):
    # split real data into train and validation

    classifier = TimeseriesClassifier(**hyperparams)
    train_classifier(classifier, train_data, val_data, epochs, verbose=False)

    classifier.eval()
    with torch.no_grad():
        predictions = classifier(test_data.sequences)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5)

    return predictions