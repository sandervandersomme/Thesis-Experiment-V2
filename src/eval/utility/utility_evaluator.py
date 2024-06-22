from src.eval.newevaluator import Evaluator


class UtilityEvaluator(Evaluator):
    def __init__(self, num_datasets: int, num_instances: int, eval_dir: str) -> None:
        super().__init__(num_datasets, num_instances, eval_dir)

    def evaluate_dataset(self, model_type: str, args):
        raise NotImplementedError