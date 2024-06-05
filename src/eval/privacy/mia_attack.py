from src.models.gen_model import GenModel
from src.models.models import select_gen_model, train_model, RWGAN
from src.training.hyperparameters import get_default_params, add_shape_to_params

# MIA stands for membership inference attack
class MIA():
    def __init__(self, device: str):
        self.device = device

    def attack():
        pass

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

    # setup GAN model
    model = select_gen_model(args.model)
    hyperparams = get_default_params(model.NAME)
    hyperparams = add_shape_to_params(hyperparams, train_data.shape)

    # Train GAN model
    model = model(**hyperparams)
    train_model(model, train_data, log_run_dir=run_dir, log_loss_dir=loss_dir)

    # Perform White Box attack
    white_box_mia = White_Box_MIA(model) 
    results = white_box_mia.attack(train_data, test_data, 0.8)
    print(results)
