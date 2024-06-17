import argparse
from src.data.data_loader import select_data
from src.data.data_processing import split_train_test
from src.models.models import select_gen_model
from src.training.hyperparameters import load_default_params, load_optimal_params
from src.training.tuning import Tuner
import torch
from src.utilities.utils import set_device
import os

class Pipeline():
    def __init__(self, args):
        self.output_path = args.output_folder
        self.build_project_folder()

        # Experiment setup
        self.task = args.task
        self.name_model = args.model
        self.name_dataset = args.dataset
        self.dataset = select_data(args.dataset)

        # Training parameters
        self.epochs = args.epochs
        self.trials = args.trials
        self.folds = args.folds
        self.seed = args.seed

        # Flags
        self.flag_gen_tuning = args.gen_tuning
        self.flag_down_tuning = args.down_tuning
        self.flag_default_params = args.default_params

        # Data
        self.train_data, self.test_data = self.process_data()
        self.train_shape = (len(self.train_data), *self.train_data[0].size())
        self.test_shape = (len(self.test_data), *self.test_data[0].size())

    def process_data(self):
        print("Preparing data..")
        if self.seed:
            generator = torch.Generator(device=set_device()).manual_seed(args.seed)
            train_data, test_data = split_train_test(self.dataset, args.split_size, generator)
        else:
            train_data, test_data = split_train_test(self.dataset, args.split_size)
        return train_data, test_data

    def execute(self):
        # tuning
        if self.flag_gen_tuning:
            print("Start tuning generative model")
            self.gen_tuning()

        if self.flag_down_tuning:
            print("Start tuning downstream model")
            self.down_tuning()
        
        # For num instances
            # Load params
            # train model
            # generate data
            # Evaluate data

        # save results

    def gen_tuning(self): 
        tuner = Tuner(self.train_data, self.name_dataset, self.seed, self.output_path)
        model = select_gen_model(self.name_model)
        tuner.tune(model, self.trials, self.folds, self.epochs)

    def down_tuning(self): 
        tuner = Tuner(self.train_data, self.name_dataset, self.seed, self.output_path)
        model = select_gen_model(self.name_model)
        tuner.tune(model, self.trials, self.folds, self.epochs)

    def train(self): pass
    def evaluate(self): pass

    def build_project_folder(self):
        os.makedirs(f"{self.output_path}syndata", exist_ok=True)
        os.makedirs(f"{self.output_path}hyperparams/trials/", exist_ok=True)
        os.makedirs(f"{self.output_path}results", exist_ok=True)
        os.makedirs(f"{self.output_path}losses", exist_ok=True)
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cf')
    parser.add_argument("--model")
    parser.add_argument("--task") 
    parser.add_argument("--gen_tuning", action="store_true")
    parser.add_argument("--down_tuning", action="store_true")
    parser.add_argument("--output_folder", default="outputs/exp1/")
    parser.add_argument("--split_size", default=0.7)
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--trials", default=10)
    parser.add_argument("--folds", default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--default_params", action="store_true")
    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.execute()