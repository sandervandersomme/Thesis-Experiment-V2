import torch
import ot
import numpy as np

from src.eval.similarity.methods_fidelity import similarity_correlation_matrix
from src.eval.visualise import visualise_longshortterm_correlations, visualise_inter_timestep_distances, visualise_tscor_similarities

def avg_diff_ts_distributions(test_data: torch.Tensor, syndata: torch.Tensor, graph_path):
    """
    For each time-step, calculates the wasserstein distance between the real and synthetic time step
    """
    test_data = test_data.numpy()
    syndata = syndata.numpy()

    # Compute wasserstein distances between real and synthetic time-steps
    sequence_length = test_data.shape[1]
    distances = {}

    for timestep in range(sequence_length):
        timestep_real = test_data[:, timestep, :]
        timestep_syn = syndata[:, timestep, :]

        cost_matrix = ot.dist(timestep_real, timestep_syn)
        distance = ot.emd2([], [], cost_matrix) 

        distances[f"timestep {timestep}"] = distance

    average_distance = np.mean(list(distances.values()))

    visualise_inter_timestep_distances(list(distances.values()), f"{graph_path}inter_ts_distances")

    return {
        "average_distance": average_distance,
    }

def avg_diff_inter_ts_distances(test_data: torch.Tensor, syndata: torch.Tensor, graph_path):
    """
    Calculates the size of the difference between real and synthetic wasserstein distance matrices
    """

    # Calculate distance matrix of real data, then of synthetic data, take difference
    diffmatrix_longshort_corrs = np.abs(inter_timestep_distances(test_data) - inter_timestep_distances(syndata))
    # Calculate magnitude of distance matrix
    frobenius_norm = np.linalg.norm(diffmatrix_longshort_corrs, 'fro')

    visualise_longshortterm_correlations(diffmatrix_longshort_corrs, f"{graph_path}long_short_term_corrs_diffs")

    return {
        "Magnitude of differences in wasserstein distances between time-steps": frobenius_norm
    }

def inter_timestep_distances(data: torch.Tensor):
    """
    Calculates the wasserstein distance matrix for each time-step pair within a dataset
    """
    data = data.numpy()

    # This function calculates the wasserstein distance matrix between time steps

    num_timesteps = data.shape[1]
    distance_matrix = np.zeros((num_timesteps, num_timesteps))

    # Loop through timesteps
    for t1 in range(num_timesteps):
        for t2 in range(t1, num_timesteps):

            # Don't calculate distance for similar timesteps
            if t1 == t2: 
                distance_matrix[t1][t2] = 0.0
                continue

            # Get timesteps: (sequences, features)
            t1_data = data[:, t1, :]
            t2_data = data[:, t2, :]

            # Calculate distance between timesteps
            cost_matrix = ot.dist(t1_data, t2_data)
            distance = ot.emd2([], [], cost_matrix)

            # Save distance in matrix
            distance_matrix[t1][t2] = distance
            distance_matrix[t2][t1] = distance

    return distance_matrix

def avg_similarity_longshortterm_correlations(test_data: torch.Tensor, syndata: torch.Tensor, columns, graph_path: str):
    """
    For each feature, calculates the difference between real and synthetic time-step correlations
    """

    # This method calculates the differences in real and synthetic correlations between timesteps 

    num_features = test_data.shape[2]

    # keep track of differences in synthetic and real timestep correlations per variable
    magnitudes = {}
    diffs_timestep_corr = {}

    # loop through variables to calculate differences in timestep correlations
    for feature_idx in range(num_features):
        # Get events of feature (sequences, events)
        real_feature_events = test_data[:, :, feature_idx]
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
        "Magnitudes of differences in real and synthetic time-step correlations per variable": np.mean(list(magnitudes.values))
    }