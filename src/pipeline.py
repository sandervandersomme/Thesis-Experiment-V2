import torch
import os
import pickle
import argparse

# Data imports
from src.data.data_loader import select_data, create_downstream_data
from src.data.data_processing import split

# Model training imports
from src.models.models import load_gen_model, train_gen_model, task_to_model
from src.training.hyperparameters import load_default_params, load_optimal_params, add_shape_to_params
from src.training.tuning import GenTuner, DownstreamTuner

# Evaluation imports
from src.eval.collector import Collector
from src.postprocessing.postprocessor import PostProcessor

class Pipeline:
    def __init__(self, args):
        self.args = args

        self.setup_paths()
        self.setup_folders()
        self.setup_data()

    def setup_paths(self):
        if self.args.output_folder is None: 
            self.ROOT_DIR = f"outputs/{self.args.dataset}/"
        else: 
            self.ROOT_DIR = f"outputs/{self.args.output_folder}/"

        self.HYPERPARAM_DIR = os.path.join(self.ROOT_DIR, f"hyperparams/")
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, f"models/")
        self.SYNDATA_DIR = os.path.join(self.ROOT_DIR, f"syndata/")
        self.EVAL_DIR = os.path.join(self.ROOT_DIR, f"eval/")
        self.RESULTS_DIR = os.path.join(self.ROOT_DIR, f"results/")


    def setup_folders(self):
        print("Setting up project folder structure...")
        os.makedirs(self.HYPERPARAM_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.SYNDATA_DIR, exist_ok=True)
        os.makedirs(self.EVAL_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

    def setup_data(self):
        print("Setting up data")
        self.real_data = select_data(self.args.dataset)
        self.train_data, self.test_data = self.split_data()
        self.train_indices = self.train_data.indices
        self.test_indices = self.test_data.indices
        self.train_shape = (len(self.train_data), *self.train_data[0].size())
        self.test_shape = (len(self.test_data), *self.test_data[0].size())

    def split_data(self):
        return split(self.real_data, self.args.split_size, self.args.seed)

    def execute(self):
        if self.args.full_cycle:            
            print("Start tuning generative models")
            self.tune_gen_models()
            print("Start tuning downstream model")
            self.tune_downstream_models()
            print("Start training models..")
            self.train_models()
            print("Start generating synthetic datasets..")
            self.generate_data()
            print("Start evaluation..")
            self.evaluate_models()
            print("Start postprocessing the results")
            self.post_process_results()
        else:
            if self.args.tune_generators:
                print("Start tuning generative models")
                self.tune_gen_models()
            if self.args.tune_downstream:
                print("Start tuning downstream model")
                self.tune_downstream_models()
            if self.args.train:
                print("Start training models..")
                self.train_models()
            if self.args.generate:
                print("Start generating synthetic datasets..")
                self.generate_data()
            if self.args.evaluate:
                print("Start evaluation..")
                self.evaluate_models()
            if self.args.postprocess:
                print("Start postprocessing the results")
                self.post_process_results()

    def tune_gen_models(self):
        for model in self.args.models:
            if self.args.dataset == "random":
                tuner = GenTuner(self.real_data, self.args.dataset, self.args.seed, self.HYPERPARAM_DIR)
            else:
                tuner = GenTuner(self.real_data[self.train_indices], self.args.dataset, self.args.seed, self.HYPERPARAM_DIR)
            tuner.tune(model, self.args.trials, self.args.epochs)

    def tune_downstream_models(self):
        train_sequences = self.real_data[self.train_data.indices]

        for task in self.args.tasks:
            if self.args.dataset == "random": train_data_downstream = create_downstream_data(self.args.dataset, task, None, None)
            else: train_data_downstream = create_downstream_data(self.args.dataset, task, train_sequences, self.real_data.columns)

            tuner = DownstreamTuner(train_data_downstream, self.args.dataset, self.args.seed, self.HYPERPARAM_DIR)
            model = task_to_model(task)
            tuner.tune(model, self.args.trials, self.args.folds, self.args.epochs)

    def train_models(self):
        for model_type in self.args.models:
            hyperparams = self.load_hyperparams(model_type)
            for instance_id in range(self.args.num_instances):
                print(f"Start training {model_type} #{instance_id}")
                model = load_gen_model(model_type, hyperparams)
                self.train_model(model)
                self.save_model(model, f"{model_type}-{instance_id}")

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
            self.save_syndata(syndata, f"{model_type}-{model_id}-{syndata_id}")

    def save_model(self, model, model_name):
        with open(os.path.join(self.MODEL_DIR, model_name + '.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_type, model_id):
        with open(os.path.join(self.MODEL_DIR, f"{model_type}-{model_id}" + '.pkl'), 'rb') as f:
            return pickle.load(f)

    def evaluate_models(self):
        self.args.real_data = self.real_data
        self.args.train_data = self.train_data
        self.args.test_data = self.test_data
        self.args.columns = self.real_data.columns

        collector = Collector(self.args.criteria, self.args.models, args.num_instances, args.num_syn_datasets, self.args, self.ROOT_DIR)
        collector.collect_results()
    
    def post_process_results(self):
        pp = PostProcessor(self.args.criteria, self.args.models, self.ROOT_DIR)
        pp.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='cf', choices=["cf", "sepsis"])
    parser.add_argument("--models", type=str, nargs='*')
    parser.add_argument("--tasks", type=str, nargs="+")
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    # Set modes
    parser.add_argument("--full_cycle", action="store_true")
    parser.add_argument("--tune_generators", action="store_true")
    parser.add_argument("--tune_downstream", action="store_true")
    parser.add_argument("--flag_default_params", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--postprocess", action="store_true")

    # Experiment settings
    parser.add_argument("--split_size", default=0.7)
    parser.add_argument("--val_split_size", default=0.15)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--trials", default=10, type=int)
    parser.add_argument("--folds", default=10, type=int)
    parser.add_argument("--num_instances", default=3, type=int)
    parser.add_argument("--num_syn_datasets", default=3, type=int)
    parser.add_argument("--num_syn_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--criteria", nargs="*", choices=["privacy", "fidelity", "utility", "diversity", "temporal", "all"])
    parser.add_argument("--n_components", type=int, default=7, help="number of components for PCA in diversity evaluation")
    parser.add_argument("--n_neighbors_diversity", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--coverage_factor", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--mia_threshold", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--dtw_threshold", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--aia_threshold", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--num_disclosed_attributes", type=int, default=5, help="number of neighbors for KNN in diversity")
    parser.add_argument("--n_neighbors_privacy", type=int, default=5, help="number of neighbors for KNN in diversity")
    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.execute()
