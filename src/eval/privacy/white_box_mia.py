from src.models.models import RWGAN
from src.models.models import GenModel, TimeGAN
import torch

class WhiteBoxMIA():
    def __init__(self, model: GenModel):
        self.device = model.device
        self.model = model

        if isinstance(model, RWGAN):
            self.discriminator = model.critic
        else:
            self.discriminator = model.discriminator

    def attack(self, train_data: torch.tensor, test_data: torch.tensor, threshold: float):
        # Configure attack data
        mixed_data = torch.cat((train_data, test_data), dim=0).to(self.device)
        train_labels = torch.ones(train_data.size(0), device=self.device)
        test_labels = torch.zeros(test_data.size(0), device=self.device)
        true_labels = torch.cat((train_labels, test_labels), dim=0)

        # Start attack
        self.discriminator.eval()

        # Get discriminator confidence scores
        with torch.no_grad():
            if isinstance(self.model, TimeGAN):
                predictions = attack_timegan(self.model, mixed_data)
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

def mia_whitebox_attack(train_data: torch.Tensor, test_data: torch.Tensor, model: GenModel, mia_threshold: float):
    mia = WhiteBoxMIA(model)
    results = mia.attack(train_data, test_data, mia_threshold)
    return results

