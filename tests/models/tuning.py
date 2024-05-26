from src.data.cf_classification import CF_Classification
from src.data.cf import CF
from src.utilities.tuning import optimize_hyperparameters

from src.models.classifier import TimeseriesClassifier, train_classifier

if __name__ == "__main__":
    # Train model on real
    cf = CF()
    cf_class = CF_Classification(cf.sequences, cf.columns.copy())
    
    best_trial = optimize_hyperparameters(cf_class, TimeseriesClassifier, train_classifier)
    print(f"Best hyperparameters: {best_trial.params}")
