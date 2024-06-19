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
        train_labels = torch.ones(train_data.size(0)).to(self.device)
        test_labels = torch.zeros(test_data.size(0)).to(self.device)
        labels = torch.cat((train_labels, test_labels), dim=0).to(self.device)

        # Start attack
        self.discriminator.eval()

        # Get discriminator confidence scores
        with torch.no_grad():
            if isinstance(self.model, TimeGAN):
                predictions = attack_timegan(self.model, mixed_data)
            else:
                predictions = self.discriminator(mixed_data)
        
        scores_vs_labels = list(zip(predictions, labels.tolist()))
        scores_vs_labels.sort(key=lambda x: x[0], reverse=True)

        train_scores = [(score.item(), label) for (score, label) in scores_vs_labels if label == 1]

        # Calculate risk
        high_confidence_scores = [score for (score, _) in train_scores if score >= threshold]
        num_samples_at_risk = len(high_confidence_scores)

        # Calculate attacker accuracy
        correct_predictions = sum(1 for (score, label) in scores_vs_labels if int(score) == label)
        total_predictions = len(scores_vs_labels)
        accuracy = correct_predictions / total_predictions

        return {
            "WBMIA samples at risk": num_samples_at_risk,
            "WBMIA samples at risk": num_samples_at_risk/len(train_data),
            "WBMIA total accuracy": accuracy
        }

def attack_timegan(model: TimeGAN, data):
    outputs = model.embedder(data)
    outputs = model.discriminator(outputs)
    return outputs

def mia_whitebox_attack(train_data, test_data, model, mia_threshold):
    mia = WhiteBoxMIA(model)
    results = mia.attack(train_data, test_data, mia_threshold)
    return results
