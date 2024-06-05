import torch
from src.eval.privacy.mia_attack import MIA
from src.models.models import GenModel, RWGAN
from src.models.models import train_model

class BlackBoxMia(MIA):
    def __init__(self, target_model: GenModel, syndata: torch.Tensor):
        super().__init__(target_model)
        shadow_model = target_model.__class__(**target_model.hyperparams)
        train_model(shadow_model, syndata)
        
        if isinstance(shadow_model, RWGAN):
            self.discriminator = shadow_model.critic
        else:
            self.discriminator = shadow_model.discriminator

def mia_blackbox_attack(train_data, test_data, model:GenModel, threshold):
    syndata = model.generate_data(200)
    mia = BlackBoxMia(model, syndata)
    results = mia.attack(train_data, test_data, threshold)
    return results

