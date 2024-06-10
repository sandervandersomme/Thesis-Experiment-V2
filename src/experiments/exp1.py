from src.training.hyperparameters import select_hyperparams
from src.data.data_loader import select_data
from src.models.models import train_model, GenModel, select_gen_model
from torch.utils.data import random_split
from src.utilities.utils import calculate_split_lengths
from src.eval.evaluator import Evaluator
from src.eval.methods import all_methods, parse_eval_arguments
import pandas as pd

import torch

if __name__ == "__main__":
    datasets = ['cf']
    models = ['rgan', 'rwgan', 'timegan']
    split = 0.8
    epochs = 50

    # Parse arguments for evaluation:
    args = parse_eval_arguments()

    for name_dataset in datasets:
        # Load and split data
        dataset = select_data(name_dataset)
        train_size, test_size = calculate_split_lengths(dataset, split)
        train_data, test_data = random_split(dataset, [train_size, test_size])
        train_data = train_data.dataset.sequences
        test_data = test_data.dataset.sequences

        # initialize evaluator
        eval = Evaluator(train_data=train_data, test_data=test_data, columns=dataset.columns, name_dataset=name_dataset, methods=all_methods, **args)

        # loop through models, train them and generate synthetic data
        for model_name in models:
            hyperparams = select_hyperparams(name_dataset, model_name, dataset.sequences.shape)
            model_class = select_gen_model(model_name)
            model: GenModel = model_class(**hyperparams)
            train_model(model, train_data, epochs)

            syndata = model.generate_data(train_size)
            torch.save(syndata, "outputs/experiments/1/syndata")
            df = pd.DataFrame(syndata.view(-1, 20)).to_csv("outputs/experiments/1/df.csv")
            eval.evaluate_dataset(syndata, model)

        # TODO: write function for getting default parameters

        # Save results
        eval.save_results("outputs/experiments/1/results.csv")
