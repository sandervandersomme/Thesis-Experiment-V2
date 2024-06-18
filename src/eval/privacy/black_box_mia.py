import torch
from src.eval.privacy.mia_attack import MIA
from src.models.models import GenModel, RWGAN
from src.models.models import train_gen_model

class BlackBoxMia(MIA):
    def __init__(self, target_model: GenModel, syndata: torch.Tensor, epochs=50):
        super().__init__(target_model)
        shadow_model = target_model.__class__(**target_model.hyperparams)
        train_gen_model(shadow_model, syndata, epochs)
        
        if isinstance(shadow_model, RWGAN):
            self.discriminator = shadow_model.critic
        else:
            self.discriminator = shadow_model.discriminator

def mia_blackbox_attack(train_data, test_data, model:GenModel, mia_threshold, epochs):
    syndata = model.generate_data(len(train_data))
    mia = BlackBoxMia(model, syndata, epochs)
    results = mia.attack(train_data, test_data, mia_threshold)
    return results

