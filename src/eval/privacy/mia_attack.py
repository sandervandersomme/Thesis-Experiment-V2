import torch
from src.models.gen_model import GenModel
from src.models.models import select_gen_model, train_model, RWGAN
from src.training.hyperparameters import get_default_params, add_shape_to_params
from src.utilities.utils import set_device

class Attack():
    def __init__(self) -> None:
        pass

class MIA(Attack):
    def __init__(self) -> None:
        super().__init__()

    def attack():
        pass

class White_Box_MIA(MIA):
    def __init__(self, model: GenModel):
        super().__init__()

        self.device = model.device

        if isinstance(model, RWGAN):
            self.discriminator = model.critic
        else:
            self.discriminator = model.discriminator

    def attack(self, train_data: torch.Tensor, test_data: torch.Tensor, threshold: 0.8):
        # Configure attack data
        mixed_data = torch.cat((train_data, test_data), dim=0).to(self.device)
        train_label = torch.ones(train_data.size(0)).to(self.device)
        test_labels = torch.zeros(test_data.size(0)).to(self.device)
        labels = torch.cat((train_label, test_labels), dim=0).to(self.device)

        # Start attack
        self.discriminator.eval()
        with torch.no_grad():
            predictions = self.discriminator(mixed_data)

        print(predictions.size())

        scores_vs_labels = list(zip(predictions, labels.tolist()))
        scores_vs_labels.sort(key=lambda x: x[0], reverse=True)

        train_scores = [(score.item(), label) for (score, label) in scores_vs_labels if label == 1]
        test_scores = [(score.item(), label) for (score, label) in scores_vs_labels if label == 0]

        print(f"High scores (likely training data): {train_scores[:10]}") 
        print(f"Low scores (likely non-training data): {test_scores[:10]}") 

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


if __name__ == "__main__":
    from src.data.random_data import generate_random_data
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, choices=["rgan", "rwgan", "timegan"], help='what model to use')
    parser.add_argument("--threshold", type=float, help='what confidence threshold to use')
    args = parser.parse_args()

    # Generate datasets
    train_data = generate_random_data(20, 5, 20)
    test_data = generate_random_data(20, 5, 20)

    # logging paths
    run_dir = None
    loss_dir = None

    # Train model
    model = select_gen_model(args.model)
    hyperparams = get_default_params(model.NAME)
    hyperparams = add_shape_to_params(hyperparams, train_data.shape)

    model = model(**hyperparams)
    train_model(model, train_data, log_run_dir=run_dir, log_loss_dir=loss_dir)

    # Perform White Box attack
    white_box_mia = White_Box_MIA(model) 
    results = white_box_mia.attack(train_data, test_data, 0.8)
    print(results)
