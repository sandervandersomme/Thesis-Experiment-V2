from src.models.classifier import TimeseriesClassifier
from src.models.models import train_classifier
from torch.utils.data import Dataset
from src.data.cf_classification import DownstreamDataset
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix
import torch
import numpy as np
from src.data.data_processing import split

def classification_scores(real_train_data: DownstreamDataset, syndata: DownstreamDataset, test_data: DownstreamDataset, epochs: int, hyperparams: dict, val_split_size: float, seed):
    train_data, val_data = split(real_train_data, val_split_size, seed)
    
    predictions_on_real = run_classifier(train_data, val_data,test_data, epochs, hyperparams, val_split_size)
    predictions_on_syn = run_classifier(syndata, val_data, test_data, epochs, hyperparams, val_split_size)
    true_labels = test_data.targets

    return {
        'Accuracy Real': accuracy_score(true_labels, predictions_on_real),
        'Accuracy Synthetic': accuracy_score(true_labels, predictions_on_syn),
        'Precision Real': precision_score(true_labels, predictions_on_real),
        'Precision Synthetic': precision_score(true_labels, predictions_on_syn),
        'Recall Real': recall_score(true_labels, predictions_on_real),
        'Recall Synthetic': recall_score(true_labels, predictions_on_syn),
        'F1 Score Real': f1_score(true_labels, predictions_on_real, zero_division=0),
        'F1 Score Synthetic': f1_score(true_labels, predictions_on_syn, zero_division=0),
        'ROC AUC Real': roc_auc_score(true_labels, predictions_on_real),
        'ROC AUC Synthetic': roc_auc_score(true_labels, predictions_on_syn),
        'Confusion Matrix Real': confusion_matrix(true_labels, predictions_on_real),
        'Diff Accuracy': np.abs(accuracy_score(true_labels, predictions_on_real) - accuracy_score(true_labels, predictions_on_syn)),
        'Diff Precision': np.abs(precision_score(true_labels, predictions_on_real) - precision_score(true_labels, predictions_on_syn)),
        'Diff Recall': np.abs(recall_score(true_labels, predictions_on_real) - recall_score(true_labels, predictions_on_syn)),
        'Diff F1 Score': f1_score(true_labels, predictions_on_real, zero_division=0) - f1_score(true_labels, predictions_on_syn, zero_division=0),
        'Diff ROC AUC': np.abs(roc_auc_score(true_labels, predictions_on_real) - roc_auc_score(true_labels, predictions_on_syn)),
        'Confusion Matrix Synthetic': confusion_matrix(true_labels, predictions_on_syn),
        'Confusion Matrix Real': confusion_matrix(true_labels, predictions_on_real)
    }
    # diffs = np.abs(metrics["Synthetic"] - metrics["Real"])
    # Test for statistical significance
    # implement bar charts for accuracy, precision, recall and f1-scores
    # Print confusion matrices

def run_classifier(train_data: Dataset, val_data: Dataset, test_data: Dataset, epochs: int, hyperparams: dict, val_split_size: float):
    # split real data into train and validation

    classifier = TimeseriesClassifier(**hyperparams)
    train_classifier(classifier, train_data, val_data, epochs)

    classifier.eval()
    with torch.no_grad():
        predictions = classifier(test_data.sequences)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5)

    return predictions