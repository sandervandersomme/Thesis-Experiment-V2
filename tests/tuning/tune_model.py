from src.data.data_loader import select_data
from src.utilities.tuning import optimize_hyperparameters

from src.models.models import TimeGAN, RGAN, RWGAN, TimeseriesClassifier, TimeseriesRegressor


if __name__ == "__main__":
    # Train model on real
    dataset = select_data("cf")
    model = RGAN
    output_path = f"outputs/testing/genmodels/"
    
    n_trials = 4

    best_trial = optimize_hyperparameters(dataset, model, output_path, n_trials)
    print(f"Best hyperparameters: {best_trial.params}")
