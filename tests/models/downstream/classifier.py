from src.models.classifier import TimeseriesClassifier, train_classifier
from src.data.cf_classification import CF_Classification
from src.models.downsteam_model import downstream_model_params
from src.data.cf import CF
import torch

if __name__ == "__main__":
    # Train model on real
    cf = CF()
    cf_class = CF_Classification(cf.sequences, cf.columns.copy())
    model = TimeseriesClassifier(cf_class.sequences.shape, **downstream_model_params)
    train_classifier(model, cf_class, f"outputs/testing/classifier/2")

    # Train model on synthetic
    syndata = torch.load(f"outputs/testing/RGAN/syn.pt")
    cf_syn = CF_Classification(syndata, cf.columns.copy())
    model2 = TimeseriesClassifier(cf_syn.sequences.shape, **downstream_model_params)
    train_classifier(model2, cf_syn, f"outputs/testing/classifier/2"
)

    


    