from src.data.data_loader import select_data

from src.models.downstream.classifier import TimeseriesClassifier, train_classifier
from src.data.cf_classification import CF_Classification
from src.models.downstream.downsteam_model import downstream_model_params

import torch
from torch.utils.data import random_split


if __name__ == "__main__":
    # Load real data
    dataset = select_data("cf")
    cf_class = CF_Classification(dataset.sequences, dataset.columns.copy())
    numseq, numev, numfeat = cf_class.sequences.shape

    # Load synthetic data
    syndata = torch.load(f"outputs/testing/genmodels/RGAN/syndata")
    cf_syn = CF_Classification(syndata, dataset.columns.copy())
    assert len(cf_class) == len(cf_syn)

    # calculate splitsizes
    train_size = int(0.8* len(cf_class))
    val_size = int(len(dataset) - train_size)
    splits = [train_size, val_size]

    # Split datasets
    train_real, val_real = random_split(cf_class, splits)
    train_syn, val_syn = random_split(cf_syn, splits)

    # Update parameters
    downstream_model_params.update({
        "num_sequences": numseq,
        "num_events": numev,
        "num_features": numfeat
    })

    # Train model on real
    model = TimeseriesClassifier(**downstream_model_params)
    model.output_path = f"outputs/testing/downstream/{model.__NAME__}/real"
    train_classifier(model, train_real, val_real, log_dir=f"runs/testing/{model.__NAME__}/real")

    # Train model on synthetic
    model2 = TimeseriesClassifier(**downstream_model_params)
    model2.output_path = f"outputs/testing/downstream/{model.__NAME__}/syn"

    train_classifier(model2, train_syn, val_syn, log_dir=f"runs/testing/{model.__NAME__}/synthetic")

    


    