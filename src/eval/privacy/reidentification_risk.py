import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


from src.data.random_data import generate_random_data
from src.data.data_transformation import flatten_into_sequences

def reidentification_risk(realdata: torch.Tensor, syndata: torch.Tensor, threshold: float, plot: bool=False):
    realdata = flatten_into_sequences(realdata)
    syndata = flatten_into_sequences(syndata)

    similarity_matrix = cosine_similarity(realdata, syndata)

    # Find the maximum similarity for each real data sample
    max_similarities = np.max(similarity_matrix, axis=1)

    # Calculate the re-identification risk
    risk = np.mean(max_similarities > threshold)

    if plot:
        sorted_similarities = np.sort(max_similarities)
        cdf = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_similarities, cdf, marker='.', linestyle='none', color='blue')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
        plt.xlabel('Maximum Similarity Scores')
        plt.ylabel('Cumulative Proportion')
        plt.title('CDF of Maximum Similarity Scores between Real and Synthetic Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    return risk

if __name__ == '__main__':
    realdata = generate_random_data(200, 5, 20)
    syndata = generate_random_data(200, 5, 20)

    risk = reidentification_risk(realdata, syndata, 0.9, True)
    print(f"reidentification risk: {risk}")
