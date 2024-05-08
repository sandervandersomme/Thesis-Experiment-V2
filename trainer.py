from typing import List

import numpy as np
import pickle

from RGAN import RGAN
from RWGAN import RWGAN
from TimeGAN import TimeGAN

classes = {
    "RGAN": RGAN,
    "RWGAN": RWGAN,
    "TimeGAN": TimeGAN
}

class Trainer():
    """
    Gets a set of model types, 
    Trains a given number of instances of that model, 
    And generates a given number of synhtetic datasets per trained instance
    """

    def __init__(self, models: Lists):
        self.models = models # List of model types to train

    def train_models(self, data, args, path):
        
        # Loop through model types
        for model in self.models:

            # create instances
            for instance_id in range(args.num_instances):
                instance = classes[model](data.shape, args.device, args.seed)

                # Train instance
                instance.train(data)

                # Save model
                model_path = f"{path}/model"
                self.save_model(instance, model_path)

                # Generate synthetic datasets
                for syn_data_id in range(args.num_datasets):
                    # Set number of samples to generate
                    samples = data.size(0) if args.num_samples is None else args.num_samples
                    syn_data = instance.generate_data(samples)
                
                    # Save datasets
                    data_path = f"{path}/{model}/syndata/{instance_id}-{syn_data_id}"
                    self.save_dataset(syn_data, data_path)

    def save_model(self, model, path):
        with open(f'{path}.pkl', 'wb') as file:
            pickle.dump(model, file)
    
    def save_dataset(self, dataset, path):
        np.save(path, dataset.numpy())




