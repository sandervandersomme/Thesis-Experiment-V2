import numpy as np
import torch

from scipy.stats import ks_2samp, skew, kurtosis
import ot

from typing import List

def stats(train_data: torch.Tensor, syndata: torch.Tensor, columns: List[str]):
    # Convert tensors to correct shape
    train_data = train_data.numpy().reshape(-1, train_data.shape[2])
    syndata = syndata.numpy().reshape(-1, syndata.shape[2])

    average_scores = {}
    var_diffs = {}
    
    # Loop through statistic methods
    for method in [np.mean, np.std, np.median, np.var, skew, kurtosis]:

        # Calculate differences and average difference in real and synthetic variable statistics
        differences = np.abs(method(train_data, axis=0) - method(syndata, axis=0))
        average_diff = np.mean(differences)

        average_scores[method.__name__] = average_diff
        var_diffs[method.__name__] = dict(zip(columns, differences))

    return {
        "average_scores": average_scores, 
        "differences per variable": var_diffs
    }

def kolmogorov_smirnov(train_data: torch.Tensor, syndata: torch.Tensor, columns: List[str]):
    "Use kolmogorov smirnov to calculate distances between variable distributions"

    # Track ks-sample test statistics and p_values
    statistics = {}
    p_values = {}

    # Loop through variable distributions
    num_features = train_data.shape[2]
    for feature_idx in range(num_features):
        real_feature = train_data[:, :, feature_idx].flatten()
        synthetic_feature = syndata[:, :, feature_idx].flatten()

        # Calculate the Kolomgorov-Smirnov two sample test
        distance, p_value = ks_2samp(real_feature, synthetic_feature)

        # Track results
        statistics[columns[feature_idx]] = distance
        p_values[columns[feature_idx]] = p_value

    average = np.mean(list(statistics.values()))

    return {
        "Average statistic": average,
        "statistics" : statistics,
        "p_values" : p_values
    }

def differences_variable_correlations(train_data: torch.Tensor, syndata: torch.Tensor, columns):
    # Calculates the difference between correlation matrices of real and synthetic data 
    # (i.e. how do correlations between variable pair differ between real and synthetic data)

    # Flatten the sequences into tabular format: (Events, number of features)
    num_features = train_data.size(2)
    real_eventlog = train_data.numpy().reshape(-1, num_features)
    synthetic_eventlog = syndata.numpy().reshape(-1, num_features)
    diff_matrix, frob_norm = diff_corr_matrix(real_eventlog, synthetic_eventlog)

    # TODO: Visualise diff_matrix

    return {
        "frob_norm": frob_norm,
        "diff_matrix": diff_matrix
    }

def wasserstein_distance(train_data: torch.Tensor, syndata: torch.Tensor, columns):
    num_features = train_data.shape[2]
    real_eventlog = train_data.numpy().reshape(-1, num_features)
    synthetic_eventlog = syndata.numpy().reshape(-1, num_features)

    cost_matrix = ot.dist(real_eventlog, synthetic_eventlog)
    distance = ot.emd2([], [], cost_matrix)
    return {"wasserstein distance": distance}

def wasserstein_distance_timesteps(train_data: torch.Tensor, syndata: torch.Tensor, columns):
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

def differences_timestep_distances(train_data: torch.Tensor, syndata: torch.Tensor, columns):
    # Calculate distance matrix of real data, then of synthetic data, take difference
    timestep_distance_matrix = np.abs(distances_timesteps(train_data) - distances_timesteps(syndata))

    # Calculate magnitude of distance matrix
    frobenius_norm = np.linalg.norm(timestep_distance_matrix, 'fro')

    # TODO: Visualise matrix

    return {
        "Magnitude of difference matrix": frobenius_norm,
        "differences in distances between timesteps": timestep_distance_matrix
    }

def distances_timesteps(data: torch.Tensor):
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

def differences_timestep_correlations(train_data: torch.Tensor, syndata: torch.Tensor, columns):
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

        diffs_correlations, frobenius_norm = diff_corr_matrix(real_feature_events, synthetic_feature_events)

        magnitudes[columns[feature_idx]] = frobenius_norm
        # Add difference correlations to dictionary
        diffs_timestep_corr[columns[feature_idx]] = diffs_correlations

    return {
        "Magnitudes": magnitudes,
        "Differences real and synthetic time-step correlations": diffs_timestep_corr
    }

def diff_corr_matrix(real_events: torch.Tensor, synthetic_events: torch.Tensor):
    # Calculate the differences between real and synthetic correlation matrices
    corr_matrix_real = np.corrcoef(real_events, rowvar=False)
    corr_matrix_syn = np.corrcoef(synthetic_events, rowvar=False)

    # Fix issue with nan-values: non-correlations
    corr_matrix_real = np.nan_to_num(corr_matrix_real)
    corr_matrix_syn = np.nan_to_num(corr_matrix_syn)

    # calculate the difference between the correlation matrices
    diff_matrix = np.abs(corr_matrix_real - corr_matrix_syn)

    # The frobenius norm calculates the size or magnitude of a matrix (in this case, the magnitude of the difference matrix)
    frobenius_norm = np.linalg.norm(diff_matrix, 'fro')

    return {"diff_matrix": diff_matrix, 
            "frobenius_norm": frobenius_norm}



# TODO: add correlations between timesteps as measure of short/long term dependencies