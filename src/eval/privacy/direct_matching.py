import torch
from sklearn.metrics import f1_score
from src.utilities.utils import set_device

def calculate_direct_matches(syndata: torch.Tensor, train_data: torch.Tensor, matching_threshold: float):
    device = set_device()

    syndata = syndata[:, None, :, :].to(device)
    t_data = train_data[None, :, :, :].to(device)
    # Use broadcasting to calculate euclidean distances between sequences
    distances = torch.sqrt(torch.sum((syndata - t_data) ** 2, dim=(2, 3))).to(device)
    inferred_membership = (distances.min(dim=0).values < matching_threshold).int().to(device)

    # Calculate F1 score
    labels = torch.ones(train_data.size(0))
    f1 = f1_score(labels.numpy(), inferred_membership.cpu())

    return {
        "f1": f1
    }


if __name__ == "__main__":
    from src.data.data_processing import generate_random_data
    syndata = generate_random_data(200,5,20)
    traindata = generate_random_data(200,5,20)

    print(calculate_direct_matches(syndata, traindata, 3.5))

