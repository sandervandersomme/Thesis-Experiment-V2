import torch
import ot
import numpy as np

from src.eval.fidelity.methods_fidelity import similarity_correlation_matrix

def similarity_event_distributions(real_data: torch.Tensor, syndata: torch.Tensor):
    """
    For each time-step, calculates the wasserstein distance between the real and synthetic time step
    """
    real_data = real_data.cpu().numpy()
    syndata = syndata.cpu().numpy()

    # Compute wasserstein distances between real and synthetic time-steps
    sequence_length = real_data.shape[1]
    var_distances = {}

    for timestep in range(sequence_length):
        timestep_real = real_data[:, timestep, :]
        timestep_syn = syndata[:, timestep, :]

        cost_matrix = ot.dist(timestep_real, timestep_syn)
        distance = ot.emd2([], [], cost_matrix) 

        var_distances[f"timestep {timestep}"] = distance

    average_distance = np.mean(list(var_distances.values()))

    return average_distance, var_distances

def similarity_temporal_distances(real_data: torch.Tensor, syndata: torch.Tensor):
    real_distances = temporal_distances(real_data)
    syn_distances = temporal_distances(syndata)
    
    difference_matrix_temporal_distances = np.abs(real_distances - syn_distances)
    
    # Apply exponential normalization to convert to a similarity matrix
    similarity_matrix = np.exp(-difference_matrix_temporal_distances)
    
    # Calculate the Frobenius norm of the similarity matrix
    frobenius_norm = np.linalg.norm(similarity_matrix, 'fro')
    
    # Normalize the Frobenius norm by the size of the matrix
    similarity_score = frobenius_norm / similarity_matrix.size
    
    return similarity_score, similarity_matrix

def temporal_distances(data: torch.Tensor):
    """
    Calculates the wasserstein distance matrix for each time-step pair within a dataset
    """
    data = data.cpu().numpy()

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

def similarity_auto_correlations(real_data: torch.Tensor, syndata: torch.Tensor, columns):
    """
    For each feature, calculates the difference between real and synthetic time-step correlations
    """

    # This method calculates the differences in real and synthetic correlations between timesteps 

    num_features = real_data.shape[2]

    # keep track of differences in synthetic and real timestep correlations per variable
    similarity_scores = {}
    temporal_correlation_matrices_per_variable = {}

    # loop through variables to calculate differences in timestep correlations
    for feature_idx in range(num_features):
        # Get events of feature (sequences, events)
        real_feature_events = real_data[:, :, feature_idx]
        # print(real_feature_events)

        synthetic_feature_events = syndata[:, :, feature_idx]

        similarities_autocorrelations, similarity_score = similarity_correlation_matrix(real_feature_events.cpu().numpy(), synthetic_feature_events.cpu().numpy())

        variable = columns[feature_idx]
        similarity_scores[variable] = similarity_score
        # Add difference correlations to dictionary
        temporal_correlation_matrices_per_variable[variable] = similarities_autocorrelations

    avg_similarity = sum(similarity_scores.values()) / len(similarity_scores)
    avg_similarity_matrix = sum(temporal_correlation_matrices_per_variable.values()) / len(temporal_correlation_matrices_per_variable)

    return avg_similarity, avg_similarity_matrix