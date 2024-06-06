import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.data.data_transformation import flatten_into_events, flatten_into_sequences

def embed_data(data, n_components=10):
    pca = PCA(n_components=n_components)
    data_embedded = pca.fit_transform(data)
    return torch.tensor(data_embedded), pca

def transform_data(data, pca):
    data_embedded = pca.transform(data)
    return torch.tensor(data_embedded)

def nearest_neighbors(data, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    return distances, nn

def calc_inter_distance(nn_real: NearestNeighbors, synthetic_data_embedded):
    distances_real_to_syn, _ = nn_real.kneighbors(synthetic_data_embedded)
    inter_set_distance = torch.mean(torch.tensor(distances_real_to_syn))
    return inter_set_distance

def calc_intra_distance(nn: NearestNeighbors, data_embedded):
    distances, _ = nn.kneighbors(data_embedded)
    intra_set_distance = torch.mean(torch.tensor(distances))
    return intra_set_distance

def calc_coverage(nn_synthetic, real_data_embedded, threshold = 1): 
    distances_syn_to_real, _ = nn_synthetic.kneighbors(real_data_embedded)
    coverage = torch.mean((torch.tensor(distances_syn_to_real) < threshold).float())
    return coverage


def calculate_diversity_scores(realdata, syndata, n_components: int, n_neighbors: int, reshape_method: str):
    if reshape_method == "sequences":
        realdata = flatten_into_sequences(realdata)
        syndata = flatten_into_sequences(syndata)
    if reshape_method == "events":
        realdata = flatten_into_events(realdata)
        syndata = flatten_into_events(syndata)
    
    # Embed data with PCA
    real_embedded, pca = embed_data(realdata, n_components)
    syn_embedded = transform_data(syndata, pca)

    # Calculate nearest neighbors of sequences
    real_distances, nn_real = nearest_neighbors(real_embedded.numpy(), n_neighbors)
    syn_distances, nn_syn = nearest_neighbors(syn_embedded.numpy(), n_neighbors)

    inter_set_distance = calc_inter_distance(nn_real, syn_embedded).item()
    intra_set_distance_syn = calc_intra_distance(nn_syn, syn_embedded).item()
    intra_set_distance_real = calc_intra_distance(nn_real, real_embedded).item()

    threshold = np.percentile(nn_syn.kneighbors(real_embedded)[0], 85)
    coverage = calc_coverage(nn_syn, real_embedded, threshold=threshold).item()

    return {
        "inter_set_distance": inter_set_distance, 
        "intra_set_distance_syn": intra_set_distance_syn, 
        "intra_set_distance_real": intra_set_distance_real,
        "normalized interset distance": inter_set_distance / intra_set_distance_real,
        "normalized intraset distance": intra_set_distance_syn / intra_set_distance_real,
        "coverage": coverage
    }



if __name__ == "__main__":
    from src.data.random_data import generate_random_data
    n_neighbors = 5
    n_components = 10

    # Generate random test data
    syndata = generate_random_data(10, 5, 20)
    realdata = generate_random_data(10, 5, 20)

    diversity_sequences = calculate_diversity_scores(realdata, syndata, n_components, n_neighbors, reshape_method="sequences")
    diversity_events = calculate_diversity_scores(realdata, syndata, n_components, n_neighbors, reshape_method="events")

    print(diversity_sequences)
    print(diversity_events)