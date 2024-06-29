import os
import torch

class Evaluator():
    def __init__(self, syndata: torch.Tensor, eval_args: str, output_dir: str) -> None:
        # Store syndata
        self.eval_args = eval_args
        self.syndata = syndata

        # Setup dirs
        self.output_dir = output_dir
        self.eval_dir = os.path.join(output_dir, "eval/")
        self.syn_data_dir = os.path.join(output_dir, "syndata/")
        self.hyperparams_dir = os.path.join(output_dir, "hyperparams/")

        # Create folder structure
        self.setup_paths()
        self.setup_folders()

    def setup_paths(self): raise NotImplementedError
    def setup_folders(self): raise NotImplementedError
    def evaluate(self, syndata: torch.Tensor): raise NotImplementedError

