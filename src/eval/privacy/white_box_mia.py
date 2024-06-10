import torch

from src.models.models import RWGAN
from src.eval.privacy.mia_attack import MIA
from src.models.models import GenModel

class WhiteBoxMIA(MIA):
    def __init__(self, model: GenModel):
        super().__init__(model)

        if isinstance(model, RWGAN):
            self.discriminator = model.critic
        else:
            self.discriminator = model.discriminator

def mia_whitebox_attack(train_data, test_data, model, mia_threshold):
    mia = WhiteBoxMIA(model)
    results = mia.attack(train_data, test_data, mia_threshold)
    return results
