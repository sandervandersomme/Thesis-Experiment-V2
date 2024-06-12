import torch
import ot
import numpy as np

from src.eval.similarity.methods_similarity import similarity_correlation_matrix
from src.eval.visualise import visualise_tscor_similarities

def wasserstein_distance_timesteps(train_data: torch.Tensor, syndata: torch.Tensor):
    """
    For each time-step, calculates the wasserstein distance between the real and synthetic time step
    """
    # Compute wasserstein distances between real and synthetic time-steps
    sequence_length = train_data.size(1)
    distances = {}

    for timestep in range(sequence_length):
        timestep_real = train_data[:, timestep, :].numpy()
        timestep_syn = syndata[:, timestep, :].numpy()

        cost_matrix = ot.dist(timestep_real, timestep_syn)
        distance = ot.emd2([], [], cost_matrix) 

        distances[f"timestep {timestep}"] = distance

    average_distance = np.mean(list(distances.values()))

    return {
        "average_distance": average_distance,
        "distances": distances
    }

def differences_timestep_distances(train_data: torch.Tensor, syndata: torch.Tensor):
    """
    Calculates the size of the difference between real and synthetic wasserstein distance matrices
    """

    # Calculate distance matrix of real data, then of synthetic data, take difference
    timestep_distance_matrix = np.abs(distances_timesteps(train_data) - distances_timesteps(syndata))

    # Calculate magnitude of distance matrix
    frobenius_norm = np.linalg.norm(timestep_distance_matrix, 'fro')

    # TODO: Visualise matrix

    return {
        "Magnitude of differences in wasserstein distances between time-steps": frobenius_norm
    }

def distances_timesteps(data: torch.Tensor):
    """
    Calculates the wasserstein distance matrix for each time-step pair within a dataset
    """

    # This function calculates the wasserstein distance matrix between time steps

    num_timesteps = data.size(1)
    distance_matrix = np.zeros((num_timesteps, num_timesteps))

    # Loop through timesteps
    for t1 in range(num_timesteps):
        for t2 in range(t1, num_timesteps):

            # Don't calculate distance for similar timesteps
            if t1 == t2: continue

            # Get timesteps: (sequences, features)
            t1_data = data[:, t1, :].numpy()
            t2_data = data[:, t2, :].numpy()

            # Calculate distance between timesteps
            cost_matrix = ot.dist(t1_data, t2_data)
            distance = ot.emd2([], [], cost_matrix) 

            # Save distance in matrix
            distance_matrix[t1][t2] = distance
            distance_matrix[t2][t1] = distance

    return distance_matrix

def differences_timestep_correlations(train_data: torch.Tensor, syndata: torch.Tensor, columns, graph_path: str):
    """
    For each feature, calculates the difference between real and synthetic time-step correlations
    """

    # This method calculates the differences in real and synthetic correlations between timesteps 

    num_features = train_data.shape[2]

    # keep track of differences in synthetic and real timestep correlations per variable
    magnitudes = {}
    diffs_timestep_corr = {}

    # loop through variables to calculate differences in timestep correlations
    for feature_idx in range(num_features):
        # Get events of feature (sequences, events)
        real_feature_events = train_data[:, :, feature_idx]
        # print(real_feature_events)

        synthetic_feature_events = syndata[:, :, feature_idx]

        sim_ts_correlations, frobenius_norm = similarity_correlation_matrix(real_feature_events, synthetic_feature_events)

        magnitudes[columns[feature_idx]] = frobenius_norm
        # Add difference correlations to dictionary
        diffs_timestep_corr[columns[feature_idx]] = sim_ts_correlations

        # Visualise diffs_timestep_corr: differences in real and synthetic time-step correlations per variable
        var_name = columns[feature_idx]
        visualise_tscor_similarities(sim_ts_correlations, f"{graph_path}sim_ts_corrs/{var_name}", var_name)

    return {
        "Magnitudes of differences in real and synthetic time-step correlations per variable": magnitudes
    }