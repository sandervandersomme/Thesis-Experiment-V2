import argparse
from typing import List
from src.data.data_loader import select_data

class Pipeline():
    def __init__(self, dataset_name: str, model_names: List[str], task: str, tuning: bool, output_path: str) -> None:
        self.dataset = select_data(dataset_name)
        self.models = model_names
        self.task = task
        self.tuning = tuning
        self.output_path = output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--models")
    parser.add_argument("--task") 
    parser.add_argument("--tuning")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    pipeline = Pipeline(args.dataset, args.models, args.task, args.tuning, args.output_path)