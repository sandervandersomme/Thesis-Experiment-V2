import torch
from src.models.models import GenModel, RWGAN, TimeGAN
from src.models.models import train_gen_model

class BlackBoxMia():
    def __init__(self, target_model: GenModel, syndata: torch.Tensor, epochs: int=50):
        self.device = target_model.device
        self.shadow_model = target_model.__class__(**target_model.hyperparams)
        train_gen_model(self.shadow_model, syndata, epochs)
        
        if isinstance(self.shadow_model, RWGAN):
            self.discriminator = self.shadow_model.critic
        else:
            self.discriminator = self.shadow_model.discriminator

    def attack(self, train_data: torch.tensor, test_data: torch.tensor, threshold: float):
        # Configure attack data
        mixed_data = torch.cat((train_data, test_data), dim=0).to(self.device)
        train_labels = torch.ones(train_data.size(0)).to(self.device)
        test_labels = torch.zeros(test_data.size(0)).to(self.device)
        true_labels = torch.cat((train_labels, test_labels), dim=0).to(self.device)

        # Start attack
        self.discriminator.eval()

        # Get discriminator confidence scores
        with torch.no_grad():
            if isinstance(self.shadow_model, TimeGAN):
                predictions = attack_timegan(self.shadow_model, mixed_data)
            else:
                predictions = self.discriminator(mixed_data)
        
        # Evaluate attack performance
        predicted_labels = (predictions >= threshold).float()

        # Calculate True Positve and False Positive rate
        tp = (predicted_labels * true_labels).sum().item()
        tn = ((1 - predicted_labels) * (1 - true_labels)).sum().item()
        fp = (predicted_labels * (1 - true_labels)).sum().item()
        fn = ((1 - predicted_labels) * true_labels).sum().item()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_accuracy_advantage = 0.5 * (tpr + (1 - fpr)) - 0.5

        return {
            "tpr": tpr, 
            "fpr": fpr, 
            "accuracy": accuracy, 
            "balanced_accuracy_advantage": balanced_accuracy_advantage 
        }


def attack_timegan(model: TimeGAN, data):
    outputs = model.embedder(data)
    outputs = model.discriminator(outputs)
    return outputs

def mia_blackbox_attack(syndata: torch.Tensor, train_data: torch.Tensor, test_data: torch.Tensor, model:GenModel, mia_threshold: float, epochs: int):
    mia = BlackBoxMia(model, syndata, epochs)
    results = mia.attack(train_data, test_data, mia_threshold)
    return results

