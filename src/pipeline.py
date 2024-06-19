import argparse
from src.data.data_loader import select_data, create_downstream_data
from src.data.data_processing import split
from src.models.models import load_gen_model, train_gen_model, task_to_model
from src.training.hyperparameters import load_default_params, load_optimal_params, add_shape_to_params
from src.training.tuning import GenTuner, DownstreamTuner
from src.eval.evaluator import Evaluator
import torch
import os
import pickle
import argparse
import os
import pickle
import torch

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.setup_folders()
        self.setup_data()
        self.HYPERPARAM_DIR = f"{self.output_path}hyperparams/trials/"
        self.MODEL_DIR = f"{self.output_path}models/"
        self.SYNDATA_DIR = f"{self.output_path}syndata/"
        self.EVAL_DIR = f"{self.output_path}eval/"

    def setup_folders(self):
        self.output_path = self.args.output_folder
        os.makedirs(f"{self.output_path}syndata", exist_ok=True)
        os.makedirs(f"{self.output_path}hyperparams/trials/", exist_ok=True)
        os.makedirs(f"{self.output_path}eval", exist_ok=True)
        os.makedirs(f"{self.output_path}losses", exist_ok=True)
        os.makedirs(f"{self.output_path}models", exist_ok=True)

    def setup_data(self):
        self.dataset = select_data(self.args.dataset)
        self.train_data, self.test_data = self.split_data()
        self.train_shape = (len(self.train_data), *self.train_data[0].size())
        self.test_shape = (len(self.test_data), *self.test_data[0].size())


    def split_data(self):
        return split(self.dataset, self.args.split_size, self.args.seed)

    def execute(self):
        if self.args.flag_gen_tuning:
            self.tune_gen_models()

        if self.args.flag_down_tuning:
            self.tune_downstream_models()

        if self.args.flag_training:
            self.train_models()

        if self.args.flag_generation:
            self.generate_data()

        if self.args.flag_evaluation:
            self.evaluate_models()

    def tune_gen_models(self):
        print("Start tuning generative models")
        for model in self.args.models:
            tuner = GenTuner(self.train_data, self.args.dataset, self.args.seed, self.output_path)
            tuner.tune(model, self.args.trials, self.args.folds, self.args.epochs)

    def tune_downstream_models(self):
        print("Start tuning downstream model")

        train_sequences = self.dataset[self.train_data.indices]
        for task in self.args.tasks:
            train_data_downstream = create_downstream_data(self.args.dataset, task, train_sequences, self.dataset.columns)

            tuner = DownstreamTuner(train_data_downstream, self.args.dataset, self.args.seed, self.output_path)
            model = task_to_model(task)
            tuner.tune(model, self.args.trials, self.args.folds, self.args.epochs)

    def train_models(self):
        for model_type in self.args.models:
            hyperparams = self.load_hyperparams(model_type)
            for instance_id in range(self.args.num_instances):
                print(f"Start training {model_type} #{instance_id}")
                model = load_gen_model(model_type, hyperparams)
                self.train_model(model)
                self.save_model(model, f"{self.args.dataset}-{model_type}-{instance_id}")

    def load_hyperparams(self, model_class):
        if self.args.flag_default_params:
            hyperparams = load_default_params(model_class)
        else:
            filename = f"{self.args.dataset}-{model_class}-{self.args.seed}"
            hyperparams = load_optimal_params(self.HYPERPARAM_DIR, filename)
        return add_shape_to_params(hyperparams, self.train_shape)

    def train_model(self, model): train_gen_model(model, self.train_data, self.args.epochs)
    
    def save_syndata(self, syndata, syndata_name): torch.save(syndata, os.path.join(self.SYNDATA_DIR, syndata_name + '.pt'))

    def generate_data(self):
        for model_type in self.args.models:
            for model_id in range(self.args.num_instances):
                model = self.load_model(model_type, model_id)

                print(f"Generating data for model {model_type} #{model_id}..")
                self.generate_datasets(model, model_type, model_id)

    def generate_datasets(self, model, model_type, model_id):
        for syndata_id in range(self.args.num_syn_datasets):
            syndata = model.generate_data(self.args.num_syn_samples)
            self.save_syndata(syndata, f"{self.args.dataset}-{model_type}-{model_id}-{syndata_id}")

    def save_model(self, model, model_name):
        with open(os.path.join(self.MODEL_DIR, model_name + '.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_type, model_id):
        with open(os.path.join(self.MODEL_DIR, f"{self.args.dataset}-{model_type}-{model_id}" + '.pkl'), 'rb') as f:
            return pickle.load(f)

    def evaluate_models(self):
        evaluator = Evaluator(self.args.criteria, self.dataset, self.train_data.indices, self.test_data.indices, self.EVAL_DIR, self.HYPERPARAM_DIR, self.args)
        for model_type in self.args.models:
            for model_id in range(self.args.num_instances):
                model = self.load_model(model_type, model_id)
                for syndata_id in range(self.args.num_syn_datasets):
                    dataset_name = f"{self.args.dataset}-{model_type}-{model_id}-{syndata_id}"
                    dataset_path = os.path.join(self.SYNDATA_DIR, dataset_name + '.pt')
                    if os.path.exists(dataset_path):
                        syndata = torch.load(dataset_path)
                        print(f"Evaluating synthetic dataset {dataset_name}")
                        evaluator.evaluate(model, syndata, dataset_name, model_type, model_id, syndata_id)
        evaluator.save_results_to_csv()
        evaluator.save_averages()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cf')
    parser.add_argument("--models", type=str, nargs='*')
    parser.add_argument("--tasks", type=str, nargs="+")
    parser.add_argument("--flag_gen_tuning", action="store_true")
    parser.add_argument("--flag_down_tuning", action="store_true")
    parser.add_argument("--flag_default_params", action="store_true")
    parser.add_argument("--flag_training", action="store_true")
    parser.add_argument("--flag_generation", action="store_true")
    parser.add_argument("--flag_evaluation", action="store_true")
    parser.add_argument("--output_folder", default="outputs/exp1/")
    parser.add_argument("--split_size", default=0.7)
    parser.add_argument("--val_split_size", default=0.15)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--trials", default=10, type=int)
    parser.add_argument("--folds", default=10, type=int)
    parser.add_argument("--num_instances", default=3, type=int)
    parser.add_argument("--num_syn_datasets", default=3, type=int)
    parser.add_argument("--num_syn_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--criteria", nargs="*", choices=["privacy", "fidelity", "utility", "diversity", "temporal_fidelity", "all"])
    parser.add_argument("--n_components", type=int, default=7, help="number of components for PCA in diversity evaluation")
    parser.add_argument("--n_neighbors_diversity", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--mia_threshold", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--matching_threshold", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--reid_threshold", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--aia_threshold", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--num_hidden_attributes", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--n_neighbors_privacy", type=int, default=5, help="number of neighbors for KNN in diversity")
    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.execute()
