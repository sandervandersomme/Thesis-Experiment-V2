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
        labels = torch.cat((train_labels, test_labels), dim=0).to(self.device)

        # Start attack
        self.discriminator.eval()

        # Get discriminator confidence scores
        with torch.no_grad():
            if isinstance(self.shadow_model, TimeGAN):
                predictions = attack_timegan(self.shadow_model, mixed_data)
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
            "BBMIA samples at risk": num_samples_at_risk,
            "BBMIA samples at risk": num_samples_at_risk/len(train_data),
            "BBMIA total accuracy": accuracy
        }

def attack_timegan(model: TimeGAN, data):
    outputs = model.embedder(data)
    outputs = model.discriminator(outputs)
    return outputs

def mia_blackbox_attack(syndata: torch.Tensor, train_data: torch.Tensor, test_data: torch.Tensor, model:GenModel, mia_threshold: float, epochs: int):
    mia = BlackBoxMia(model, syndata, epochs)
    results = mia.attack(train_data, test_data, mia_threshold)
    return results

