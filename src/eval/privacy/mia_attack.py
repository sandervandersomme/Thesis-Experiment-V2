import torch

from src.models.gen_model import GenModel
from src.models.models import select_gen_model, train_model, RWGAN, TimeGAN
from src.training.hyperparameters import get_default_params, add_shape_to_params

# MIA stands for membership inference attack
class MIA():
    def __init__(self, model: GenModel):
        self.device = model.device
        self.discriminator = None
        self.model = model

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
            "total training samples": len(train_data),
            "samples at risk": num_samples_at_risk,
            "samples at risk": num_samples_at_risk/len(train_data),
            "total accuracy": accuracy
        }

def attack_timegan(model: TimeGAN, data):
    outputs = model.embedder(data)
    outputs = model.discriminator(data)
    return outputs

if __name__ == "__main__":
    from src.data.random_data import generate_random_data
    from src.eval.privacy.white_box_mia import WhiteBoxMIA
    from src.eval.privacy.black_box_mia import BlackBoxMia
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["rgan", "rwgan", "timegan"], help='what model to use')
    parser.add_argument("--threshold", type=float, help='what confidence threshold to use')
    args = parser.parse_args()

    # Generate datasets
    train_data = generate_random_data(20, 5, 20)
    test_data = generate_random_data(20, 5, 20)

    # setup GAN model
    model: GenModel = select_gen_model(args.model)
    hyperparams = get_default_params(model.NAME, train_data.shape)
    hyperparams = add_shape_to_params(hyperparams, train_data.shape)

    # Train GAN model
    model = model(**hyperparams)
    train_model(model, train_data)

    # Perform White Box attack
    # white_box_mia = WhiteBoxMIA(model) 
    # results = white_box_mia.attack(train_data, test_data, 0.5)
    # print(results)

    # Perform Black Box attack
    syndata = model.generate_data(200)
    black_box_mia = BlackBoxMia(model, syndata) 
    results = black_box_mia.attack(train_data, test_data, 0.5)
    print(results)
