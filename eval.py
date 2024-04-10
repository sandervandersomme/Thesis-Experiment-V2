import torch
from utils import load_data, load_synthetic_data, split_train_test, visualise
import argparse
from eval_utility import ClassifierRNN, classify_real_synthetic
from eval_similarity import difference_statistics, univariate_distances, bivariate_distances, multivariate_distances
from typing import List

class Evaluator():
    def __init__(self, real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str], path):
        assert real_data.size(0) == synthetic_data.size(0)
        assert real_data.size(1) == synthetic_data.size(1)
        assert real_data.size(2) == synthetic_data.size(2)
        assert real_data.size(2) == len(columns)

        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.columns = columns
        self.path = path

    def evaluate_similarity(self):
        print("Evaluating similarity..")
        # stats = difference_statistics(self.real_data, self.synthetic_data)
        # distances_univariate = univariate_distances(self.real_data, self.synthetic_data, self.columns, self.path)
        # distances_bivariate = bivariate_distances(self.real_data, self.synthetic_data)
        distances_multivariate = multivariate_distances(self.real_data, self.synthetic_data, self.path)

    def evaluate_privacy(self):
        print("Evaluating privacy..")

    def evaluate_utility(self, epochs, learning_rate):
        print("Evaluating utility..")

        ## Train and evaluate classifier to distinguish between real and synthetic data
        train_data, test_data = split_train_test(self.real_data, self.synthetic_data)
        classify_real_synthetic()




def parse_args():
    print(f"Parsing arguments..")
    parser = argparse.ArgumentParser()
    parser.add_argument('--realdata', type=str, help='What dataset should be used?', default='dummy', choices=['cf', 'sepsis', 'dummy'])
    parser.add_argument('--syndata', type=str, help='What is the name of the synthetic dataset (dataset-epochs)')
    parser.add_argument('--model', type=str, help='What model should be used?', choices=['RGAN', 'TimeGAN'])
    parser.add_argument('--epochs', type=int, help='Sets the number of epochs for training', default=10)
    parser.add_argument('--epochid', type=int, help='To identify the synthetic dataset', default=None)
    parser.add_argument('--learning_rate', type=float, help="Learning rate for the optimizer of the classifier", default=0.001)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments for data evaluation
    args = parse_args()

    # Load real and synthetic datasets, split into training and testing datasets
    real_data, columns = load_data(args.realdata)
    synthetic_data = load_synthetic_data(args.syndata, args.model)
    
    # Create evaluator object
    evaluator = Evaluator(real_data, synthetic_data, columns, args.syndata)

    # Evaluate synthetic data
    evaluator.evaluate_similarity()
    # evaluator.evaluate_utility(args.epochs, args.learning_rate)
    # evaluator.evaluate_privacy()

    print("Done!")
