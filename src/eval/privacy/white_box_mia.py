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

    def attack(self, train_data: torch.Tensor, test_data: torch.Tensor, threshold: float):
        # Configure attack data
        mixed_data = torch.cat((train_data, test_data), dim=0).to(self.device)
        train_label = torch.ones(train_data.size(0)).to(self.device)
        test_labels = torch.zeros(test_data.size(0)).to(self.device)
        labels = torch.cat((train_label, test_labels), dim=0).to(self.device)

        # Start attack
        self.discriminator.eval()
        with torch.no_grad():
            predictions = self.discriminator(mixed_data)

        scores_vs_labels = list(zip(predictions, labels.tolist()))
        scores_vs_labels.sort(key=lambda x: x[0], reverse=True)

        train_scores = [(score.item(), label) for (score, label) in scores_vs_labels if label == 1]
        test_scores = [(score.item(), label) for (score, label) in scores_vs_labels if label == 0]

        # print(f"High scores (likely training data): {train_scores[:10]}") 
        # print(f"Low scores (likely non-training data): {test_scores[:10]}") 

        # Calculate risk
        high_confidence_scores = [score for (score, label) in train_scores if score >= threshold]
        num_samples_at_risk = len(high_confidence_scores)

        # Calculate attacker accuracy
        correct_predictions = sum(1 for (score, label) in scores_vs_labels if score == label)
        total_predictions = len(labels)
        accuracy = correct_predictions / total_predictions

        return {
            "total training samples": len(train_data),
            "samples at risk": num_samples_at_risk,
            "samples at risk": num_samples_at_risk/len(train_data),
            "total accuracy": accuracy
        }

def mia_whitebox_attack(train_data, test_data, model, threshold):
    mia = WhiteBoxMIA(model)
    results = mia.attack(train_data, test_data, threshold)
    return results
