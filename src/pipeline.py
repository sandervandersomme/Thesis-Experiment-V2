import argparse
from src.data.data_loader import select_data, create_downstream_data
from src.data.data_processing import split_train_test
from src.models.models import load_gen_model, load_downstream_model, train_gen_model, train_downstream_model, GenModel, task_to_model
from src.training.hyperparameters import load_default_params, load_optimal_params, add_shape_to_params
from src.training.tuning import GenTuner, DownstreamTuner
from src.eval.evaluator import Evaluator
import torch
from src.utilities.utils import set_device
import os
import pickle

class Pipeline():
    def __init__(self, args):
        self.output_path = args.output_folder
        self.build_project_folder()

        # pipeline setup
        self.task = args.task
        self.name_dataset = args.dataset
        self.models = args.models
        self.num_instances = args.num_instances
        self.num_syn_samples = args.num_syn_samples
        self.num_syn_datasets = args.num_syn_datasets

        # Training parameters
        self.epochs = args.epochs
        self.trials = args.trials
        self.folds = args.folds
        self.seed = args.seed

        # Flags
        self.flag_gen_tuning = args.flag_gen_tuning
        self.flag_down_tuning = args.flag_down_tuning
        self.flag_default_params = args.flag_default_params
        self.flag_training = args.flag_training
        self.flag_generation = args.flag_generation
        self.flag_evaluation = args.flag_evaluation

        # Data
        self.dataset = select_data(args.dataset)
        self.train_data, self.test_data = self.process_data()
        self.train_shape = (len(self.train_data), *self.train_data[0].size())
        self.test_shape = (len(self.test_data), *self.test_data[0].size())

        if self.task:
            train_sequences = self.dataset[self.train_data.indices]
            test_sequences = self.dataset[self.test_data.indices]
            self.train_data_downstream = create_downstream_data(self.name_dataset, self.task, train_sequences, self.dataset.columns, f"train_real_{self.task}")
            self.test_data_downstream = create_downstream_data(self.name_dataset, self.task, train_sequences, self.dataset.columns, f"test_real_{self.task}")

        self.HYPERPARAM_DIR = f"{self.output_path}hyperparams/trials/"
        self.MODEL_DIR = f"{self.output_path}models/"
        self.SYNDATA_DIR = f"{self.output_path}syndata/"

    def process_data(self):
        print("Preparing data..")
        if self.seed:
            generator = torch.Generator(device=set_device()).manual_seed(args.seed)
            train_data, test_data = split_train_test(self.dataset, args.split_size, generator)
        else:
            train_data, test_data = split_train_test(self.dataset, args.split_size)
        return train_data, test_data

    def execute(self):
        if self.flag_gen_tuning:
            print("Start tuning generative model")
            for model in self.models:
                self.gen_tuning(model)

        if self.flag_down_tuning:
            print("Start tuning downstream model")
            self.down_tuning()

        # Model training
        if self.flag_training:
            # Loop through model types
            for model_type in self.models:

                # Load hyperparams
                hyperparams = self.load_params(model_type)

                # Train multiple instances
                for instance_id in range(self.num_instances):
                    model = load_gen_model(model_type, hyperparams)
                    self.train(model)
                    self.save_model(model, f"{self.name_dataset}-{model_type}-{instance_id}")

                    # Generate multiple synthetic datasets
                    for syndata_id in range(self.num_syn_datasets):
                        syndata = self.generate(model)
                        self.save_syndata(syndata, f"{self.name_dataset}-{model_type}-{instance_id}-{syndata_id}")

        if self.flag_evaluation:
            self.evaluate()

        # save results

    def gen_tuning(self, model_class): 
        tuner = GenTuner(self.train_data, self.name_dataset, self.seed, self.output_path)
        tuner.tune(model_class, self.trials, self.folds, self.epochs)

    def down_tuning(self):
        tuner = DownstreamTuner(self.train_data_downstream, self.name_dataset, self.seed, self.output_path)
        model = task_to_model(self.task)
        tuner.tune(model, self.trials, self.folds, self.epochs)

    def load_params(self, model_class: str):
        if self.flag_default_params: hyperparams = load_default_params(model_class)
        else: 
            filename = f"{self.name_dataset}-{model_class}-{self.seed}"
            hyperparams = load_optimal_params(self.HYPERPARAM_DIR, filename)
        hyperparams = add_shape_to_params(hyperparams, self.train_shape)
        return hyperparams

    def evaluate(self):
        results = {}
        evaluator = Evaluator()

        # Select criteria

        # Loop through all synthetic datasets
        for model_type in self.models:
            for model_id in range(self.num_instances):
                for syndata_id in range(self.num_syn_datasets):
                    dataset_name = f"{self.name_dataset}-{model_type}-{model_id}-{syndata_id}"
                    dataset_path = os.path.join(self.SYNDATA_DIR, dataset_name + '.pt')

                    # Load dataset
                    if os.path.exists(dataset_path):
                        syndata = torch.load(dataset_path)

                        # Evaluate dataset
                        evaluator.evaluate(syndata, dataset_name, model_type, model_id, syndata_id)
                        

    def train(self, model: GenModel): train_gen_model(model, self.train_data, self.epochs)        
    def generate(self, model: GenModel): model.generate_data(self.num_syn_samples)
    def save_model(self, model, model_name): pickle.dump(model, open(os.path.join(self.MODEL_DIR, model_name + '.pkl'), 'wb'))
    def save_syndata(self, syndata, syndata_name): torch.save(syndata, os.path.join(self.SYNDATA_DIR, syndata_name + '.pt'))

    def build_project_folder(self):
        os.makedirs(f"{self.output_path}syndata", exist_ok=True)
        os.makedirs(f"{self.output_path}hyperparams/trials/", exist_ok=True)
        os.makedirs(f"{self.output_path}eval", exist_ok=True)
        os.makedirs(f"{self.output_path}losses", exist_ok=True)
        os.makedirs(f"{self.output_path}models", exist_ok=True)

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cf')
    parser.add_argument("--models", type=str, nargs='*')
    parser.add_argument("--task", type=str)
    parser.add_argument("--flag_gen_tuning", action="store_true")
    parser.add_argument("--flag_down_tuning", action="store_true")
    parser.add_argument("--flag_default_params", action="store_true")
    parser.add_argument("--flag_training", action="store_true")
    parser.add_argument("--flag_generation", action="store_true")
    parser.add_argument("--flag_evaluation", action="store_true")
    parser.add_argument("--output_folder", default="outputs/exp1/")
    parser.add_argument("--split_size", default=0.7)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--trials", default=10, type=int)
    parser.add_argument("--folds", default=10, type=int)
    parser.add_argument("--num_instances", default=3, type=int)
    parser.add_argument("--num_syn_datasets", default=3, type=int)
    parser.add_argument("--num_syn_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.execute()

