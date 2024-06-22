from src.eval.newevaluator import Evaluator


class TemporalFidelityEvaluator(Evaluator):
    def __init__(self, num_datasets: int, num_instances: int, eval_dir: str) -> None:
        super().__init__(num_datasets, num_instances, eval_dir)

    def evaluate_dataset(self, dataset, dataset_id, model, model_id):
        raise NotImplementedError