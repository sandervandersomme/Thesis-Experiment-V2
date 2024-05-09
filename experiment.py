from trainer import Trainer
from evaluator import Evaluator

from utils import parse_args
from data_utils import load_data

import os
import pickle
import numpy as np
from typing import List

INPUT_PATH = f"data/input"

datasets = ["cf", "cf_subset_time"]
models = ["RGAN"]
metrics = ["similarity", "utility", "privacy", "diversity", "fairness"]

class Experiment:
    def __init__(self, model_classes: List, datasets: List, metrics: List, experiment_id: int):
        self.model_classes = model_classes
        self.datasets = datasets
        self.metrics = metrics

        self.id = experiment_id
        self.OUTPUT_PATH = f"results/experiment-{self.id}"

    def run(self, args):
        # Create folder structure
        create_output_structure(self.OUTPUT_PATH)

        # Create evaluator
        evaluator = Evaluator(metrics)

        # Generate datasets
        for dataset in self.datasets:
            data, columns = load_data(f"{dataset}.csv")

            # Loop through model types
            for model_class in self.model_classes:
                
                # Generate instances per model
                for instance_id in range(args.num_instances):
                    instance = self.model_classes[model_class](data.shape, args.device, args.seed)

                    # Train instance
                    instance.train(data)

                    # Save model
                    model_path = f"{self.__OUTPUT_PATH__}/{dataset}/{model_class}"
                    save_model(instance, model_path)

                    # Generate synthetic datasets
                    for syndata_id in range(args.num_datasets):

                        # Set number of samples to generate
                        samples = data.size(0) if args.num_samples is None else args.num_samples
                        syn_data = instance.generate_data(samples)
                    
                        # Save datasets
                        data_path = f"{model_path}/syndata/{instance_id}-{syndata_id}"
                        save_dataset(syn_data, data_path)

                        # Evaluate dataset
                        results = evaluator.evaluate(syn_data)



def create_output_structure(path):
    for dataset in datasets:
        for model in models:
            if not os.path.exists(f"{path}/{dataset}/{model}"):
                os.makedirs(f"{path}/{dataset}/{model}/syndata")

def save_model(model, path):
    with open(f'{path}.pkl', 'wb') as file:
        pickle.dump(model, file)

def save_dataset(dataset, path):
    np.save(path, dataset.numpy())