import argparse
from datetime import datetime
import torch

# Set paths
from src.paths import PATH_EXP1_UNTUNED

# Import models and data
from src.models.models import select_model, train_model
from src.models.gen_model import GenModel
from src.data.data_loader import select_data

from src.training.hyperparameters import get_default_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to be used', default='cf')
    parser.add_argument('--model', type=str, help='Model type to be tuned')
    args = parser.parse_args()

    # Setup experiment
    dataset = select_data(args.dataset)
    model_class = select_model(args.model)
    hyperparams = get_default_params(args.model)

    # Train model
    model: GenModel = model_class(**hyperparams)
    model.NAME += "-untuned"
    moment = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_run_dir = PATH_EXP1_UNTUNED + f"runs/{dataset.NAME}-{model.NAME}/{moment}"
    log_loss_dir = PATH_EXP1_UNTUNED + f"losses/{dataset.NAME}-{model.NAME}/{moment}"
    train_model(model, dataset, log_run_dir, log_loss_dir)

    syndata: torch.Tensor = model.generate_data(len(dataset))
    torch.save(syndata, PATH_EXP1_UNTUNED + f"{dataset.NAME}-{model.NAME}/{moment}")


    
