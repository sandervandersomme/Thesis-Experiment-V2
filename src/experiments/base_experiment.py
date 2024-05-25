import os
import pickle
import numpy as np
from torch.utils.data import DataLoader

from src.utilities.utils import set_device

from src.data.data_loader import get_data

INPUT_PATH = f"data/input"

class Experiment:
    def __init__(self, seed, **settings):
        # Configurations of experiment
        self.models = settings["models"]
        self.metrics = settings["metrics"]
        self.dataset = settings["dataset"]
        self.num_instances = settings["num_instances"], 
        self.num_datasets = settings["num_datasets"], 
        self.train_size = settings["train_size"]

        # Torch settings
        self.device = set_device()
        self.seed = seed

        create_output_structure(settings["models"], settings["dataset"], settings["exp_id"])

def run(self, tune=False):
        # Load data
        train_data, _ = get_data(self.dataset, self.task, self.train_size)
        num_features = train_data[0].shape[1]

        # Loop through model types
        for model in self.models:

            if tune:
                hyperparams = tune_model(model.__MODEL__)
            else: 
                hyperparams = load_params(model.__MODEL__)

            # Generate instances per model
            for instance_id in range(self.num_instances):
                instance = model(self.device, self.seed, num_features=num_features, **hyperparams)

                # Train instance
                instance.train(train_data)

                # Save model
                model_path = f"results/experiment-{self.id}/{self.dataset}/{model.__MODEL__}"
                save_model(instance, model_path)

                # Generate synthetic datasets
                for syndata_id in range(self.num_datasets):

                    # Set number of samples to generate
                    samples = real_data.size(0) if self.args["num_samples"] is None else self.args["num_samples"]
                    synthetic_data = instance.generate_data(samples)
                
                    # Save datasets
                    data_path = f"{model_path}/syndata/{instance_id}-{syndata_id}"
                    save_dataset(synthetic_data, data_path)

                    # Evaluate dataset
                    results = evaluator.evaluate(real_data=real_data, synthetic_data=synthetic_data, columns=columns, task=self.task)

                    with open(f"{model_path}/scores/{instance_id}-{syndata_id}", 'w') as json_file:
                        json.dump(results, json_file, indent=4, default=handle_numpy)

class Classification_Experiment(Experiment):
    def __init__(self, **settings):
        self.task = "classification"
        super().__init__(**settings)

    



def create_output_structure(models, dataset: str, experiment_id):
    for model in models:
        path = f"results/experiment-{experiment_id}/{dataset}/{model.__MODEL__}"
        if not os.path.exists(path):
            os.makedirs(f"{path}/syndata")
            os.makedirs(f"{path}/scores")

def save_model(model, path):
    with open(f'{path}.pkl', 'wb') as file:
        pickle.dump(model, file)

def save_dataset(dataset, path):
    np.save(path, dataset.numpy())