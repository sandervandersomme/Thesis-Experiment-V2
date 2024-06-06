import torch

from torch.utils.tensorboard import SummaryWriter

from src.training.early_stopping import EarlyStopping
from src.utilities.utils import set_device

class Trainer():
    def __init__(self, model, train_data: torch.Tensor, validate=False, logging_path=None, loss_path=None, device=set_device(), stop_early=True, verbose=False):
        # set logging bool
        self.model = model
        self.train_data = train_data
        self.device = device

        self.stop_early = stop_early
        self.validate = validate
        self.verbose = verbose
        self.writer = SummaryWriter(logging_path) if logging_path else None
        self.loss_path = loss_path

        # setup early stopping 
        if stop_early: self.early_stopping = EarlyStopping(model.patience, model.min_delta)
        if validate or stop_early:
            self.best_val_loss = float('inf')
            self.val_losses = []

        self.epoch = 0

    def train(self):
        pass

    def validation(self, val_loss):
        # Check if best loss has increased
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if self.writer:
            self.writer.add_scalar('Loss/val', val_loss, self.epoch)
