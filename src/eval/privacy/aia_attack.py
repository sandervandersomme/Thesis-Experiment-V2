import torch
import numpy as np
from typing import List
from sklearn.metrics import f1_score

# AIA stands for Attribute Inference Attack
class AIA:
    def __init__(self, known_indices: List[int], unknown_indices: List[int], k: int = 1, threshold: float = 0.1):
        self.known_indices = known_indices
        self.unknown_indices = unknown_indices
        self.k = k
        self.threshold = threshold

    def find_knn(self, target_sequence: torch.Tensor, syndata: torch.Tensor) -> torch.Tensor:
        distances = torch.norm(syndata[:, :, self.known_indices] - target_sequence[:, self.known_indices].unsqueeze(0), dim=(1, 2))
        neighbors_indices = torch.argsort(distances)[:self.k]
        return neighbors_indices

    def majority_rule_classifier(self, neighbors: torch.Tensor, unknown_index: int) -> torch.Tensor:
        unique, counts = torch.unique(neighbors[:, :, unknown_index], return_counts=True)
        return unique[torch.argmax(counts)]

    def calculate_risk(self, syndata: torch.Tensor, realdata: torch.Tensor) -> float:
        inferred_attributes = []
        true_attributes = realdata[:, :, self.unknown_indices]

        for sequence in realdata:
            neighbors_indices = self.find_knn(sequence, syndata)
            neighbors = syndata[neighbors_indices]

            inferred_sequence = []
            for event in sequence:
                inferred_event = [self.majority_rule_classifier(neighbors, index) for index in self.unknown_indices]
                inferred_sequence.append(inferred_event)
            inferred_attributes.append(inferred_sequence)

        inferred_attributes = torch.tensor(inferred_attributes)

        # calculate weights
        information_entropy = -torch.sum(realdata * torch.log2(realdata + 1e-9), dim=(0, 1))
        weights = information_entropy / torch.sum(information_entropy)
        weigths = weights[self.unknown_indices]

        weighted_f1_score = 0.0
        weighted_accuracy = 0.0

        for i in range(true_attributes.shape[2]):
            if is_binary(true_attributes[:, :, i]): # Binary attribute
                f1_score = f1_score(true_attributes[:, :, i].flatten().cpu().numpy(), inferred_attributes[:, :, i].flatten().cpu().numpy())
                weighted_f1_score += weights[i] * f1_score
            else:  # Continuous attribute
                weighted_accuracy += weights[i] * torch.mean((torch.abs(true_attributes[:, :, i] - inferred_attributes[:, :, i]) < self.threshold).float()).item()

        attribute_inference_risk = weighted_f1_score + weighted_accuracy

        return attribute_inference_risk

def is_binary(attribute): return torch.unique(attribute).numel() == 2

def perform_aia(syndata: torch.Tensor, traindata: torch.Tensor, unknown_indices:torch.Tensor=None, k=1, threshold=0.1, num_disclosed_attributes:int=3):
    if unknown_indices is None:
        unknown_indices = torch.randperm(realdata.size(2))[:num_disclosed_attributes]
        known_indices = get_known_indices(unknown_indices, realdata.size(2))
    else:
        known_indices = known_indices(unknown_indices)

    attack = AIA(known_indices, unknown_indices, k=1, threshold=0.1)
    risk = attack.calculate_risk(syndata, traindata)

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