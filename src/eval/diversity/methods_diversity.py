import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.data.data_processing import flatten_into_events, flatten_into_sequences

def embed_data(data: torch.Tensor, n_components=10):
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

def calc_distance(nn: NearestNeighbors, data):
    distances, _ = nn.kneighbors(data)
    set_distance = torch.mean(torch.tensor(distances))
    return set_distance

def calc_coverage(nn_synthetic, real_data_embedded, threshold = 1): 
    distances_syn_to_real, _ = nn_synthetic.kneighbors(real_data_embedded)
    coverage = torch.mean((torch.tensor(distances_syn_to_real) < threshold).float())
    return coverage

def calculate_diversity_scores(train_data: torch.Tensor, syndata: torch.Tensor, n_components: int, n_neighbors: int, reshape_method: str="events"):
    if reshape_method == "sequences":
        train_data = flatten_into_sequences(train_data)
        syndata = flatten_into_sequences(syndata)
    if reshape_method == "events":
        train_data = flatten_into_events(train_data)
        syndata = flatten_into_events(syndata)
    
    # Embed data with PCA
    real_embedded, pca = embed_data(train_data, n_components)
    syn_embedded = transform_data(syndata, pca)

    # Calculate nearest neighbors of sequences
    real_distances, nn_real = nearest_neighbors(real_embedded.numpy(), n_neighbors)
    syn_distances, nn_syn = nearest_neighbors(syn_embedded.numpy(), n_neighbors)

    inter_set_distance = calc_distance(nn_real, syn_embedded).item()
    intra_set_distance_syn = calc_distance(nn_syn, syn_embedded).item()
    intra_set_distance_real = calc_distance(nn_real, real_embedded).item()

    relative_diversity = intra_set_distance_syn - intra_set_distance_real
    relative_coverage = inter_set_distance - intra_set_distance_real

    threshold = np.percentile(nn_syn.kneighbors(real_embedded)[0], 85)
    absolute_coverage = calc_coverage(nn_syn, real_embedded, threshold=threshold).item()

    return absolute_coverage, relative_coverage, relative_diversity

