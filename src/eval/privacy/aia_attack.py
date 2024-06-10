import torch
import numpy as np
from typing import List
from sklearn.metrics import f1_score

# AIA stands for Attribute Inference Attack
class AIA:
    def __init__(self, known_indices: List[int], unknown_indices: List[int], aia_threshold: float = 0.1,k: int = 1):
        self.known_indices = known_indices
        self.unknown_indices = unknown_indices
        self.k = k
        self.threshold = aia_threshold

    def find_knn(self, target_sequence: torch.Tensor, syndata: torch.Tensor) -> torch.Tensor:
        distances = torch.norm(syndata[:, :, self.known_indices] - target_sequence[:, self.known_indices].unsqueeze(0), dim=(1, 2))
        neighbors_indices = torch.argsort(distances)[:self.k]
        return neighbors_indices

    def majority_rule_classifier(self, neighbors: torch.Tensor, unknown_index: int) -> torch.Tensor:
        unique, counts = torch.unique(neighbors[:, :, unknown_index], return_counts=True)
        return unique[torch.argmax(counts)]

    def calculate_risk(self, syndata: torch.Tensor, real_data: torch.Tensor) -> float:
        inferred_attributes = []
        true_attributes = real_data[:, :, self.unknown_indices]

        for sequence in real_data:
            neighbors_indices = self.find_knn(sequence, syndata)
            neighbors = syndata[neighbors_indices]

            inferred_sequence = []
            for event in sequence:
                inferred_event = [self.majority_rule_classifier(neighbors, index) for index in self.unknown_indices]
                inferred_sequence.append(inferred_event)
            inferred_attributes.append(inferred_sequence)

        inferred_attributes = torch.tensor(inferred_attributes)

        # calculate weights
        information_entropy = -torch.sum(real_data * torch.log2(real_data + 1e-9), dim=(0, 1))
        weights = information_entropy / torch.sum(information_entropy)
        weights = weights[self.unknown_indices]

        weighted_f1_score = 0.0
        weighted_accuracy = 0.0

        print(self.unknown_indices)
        for i in range(true_attributes.shape[2]):
            if is_binary(true_attributes[:, :, i]): # Binary attribute
                print("is binary")
                print(true_attributes[:,:,i])
                print(inferred_attributes[:,:,i])

                true_flat = true_attributes[:, :, i].flatten().cpu().numpy()
                inferred_flat = inferred_attributes[:, :, i].flatten().cpu().numpy()
                true_binary = np.round(true_flat).astype(int)
                inferred_binary = np.round(inferred_flat).astype(int)

                f1 = f1_score(true_binary, inferred_binary)
                weighted_f1_score += weights[i] * f1
            else:  # Continuous attribute
                print("is continuous")

                accuracy = torch.mean((torch.abs(true_attributes[:, :, i] - inferred_attributes[:, :, i]) < self.threshold).float()).item()
                weighted_accuracy += weights[i] * accuracy

        attribute_inference_risk = weighted_f1_score + weighted_accuracy

        return attribute_inference_risk

def is_binary(tensor): return torch.all((tensor == 0) | (tensor == 1))

def perform_aia(syndata: torch.Tensor, train_data: torch.Tensor, k=1, aia_threshold=0.1, num_disclosed_attributes:int=3):
    unknown_indices = torch.randperm(train_data.size(2))[:num_disclosed_attributes]
    known_indices = get_known_indices(unknown_indices, train_data.size(2))

    attack = AIA(known_indices, unknown_indices, aia_threshold, k=1)
    risk = attack.calculate_risk(syndata, train_data)

    return {
        "risk": risk
    }

def get_known_indices(unknown_indices:torch.Tensor, num_features):
    # Determine random attributes to hide
    all_indices = torch.arange(num_features)
    mask = torch.ones(num_features, dtype=bool)
    mask[unknown_indices] = 0
    known_indices = all_indices[mask]

    return known_indices

if __name__ == "__main__":
    from src.data.random_data import generate_random_data

    # Example usage
    syndata = generate_random_data(400,5,20)
    realdata = generate_random_data(400,5,20)

    # Define the indices for known and unknown attributes
    risk = perform_aia(syndata, realdata)
    print(risk)