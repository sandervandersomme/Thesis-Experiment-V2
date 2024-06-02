import numpy as np

from scipy.stats import ks_2samp, skew, kurtosis
import ot

from typing import List

def stats(real_data: np.array, synthetic_data: np.array, columns: List[str]):
    # Convert tensors to correct shape
    real_data = real_data.reshape(-1, real_data.shape[2])
    synthetic_data = synthetic_data.reshape(-1, synthetic_data.shape[2])

    average_scores = {}
    var_diffs = {}
    
    # Loop through statistic methods
    for method in [np.mean, np.std, np.median, np.var, skew, kurtosis]:

        # Calculate differences and average difference in real and synthetic variable statistics
        differences = np.abs(method(real_data, axis=0) - method(synthetic_data, axis=0))
        average_diff = np.mean(differences)

        average_scores[method.__name__] = average_diff
        var_diffs[method.__name__] = dict(zip(columns, differences))

    return {
        "average_scores": average_scores, 
        "differences per variable": var_diffs
    }

def kolmogorov_smirnov(real_data: np.array, synthetic_data: np.array, columns: List[str]):
    "Use kolmogorov smirnov to calculate distances between variable distributions"

    # Track ks-sample test statistics and p_values
    statistics = {}
    p_values = {}

    # Loop through variable distributions
    num_features = real_data.shape[2]
    for feature_idx in range(num_features):
        real_feature = real_data[:, :, feature_idx].flatten()
        synthetic_feature = synthetic_data[:, :, feature_idx].flatten()

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

def differences_variable_correlations(real_data: np.array, synthetic_data: np.array, columns):
    # Calculates the difference between correlation matrices of real and synthetic data 
    # (i.e. how do correlations between variable pair differ between real and synthetic data)

    # Flatten the sequences into tabular format: (Events, number of features)
    num_features = real_data.shape[2]
    real_eventlog = real_data.reshape(-1, num_features)
    synthetic_eventlog = synthetic_data.reshape(-1, num_features)
    diff_matrix, frob_norm = diff_corr_matrix(real_eventlog, synthetic_eventlog)

    # TODO: Visualise diff_matrix

    return {
        "frob_norm": frob_norm,
        "diff_matrix": diff_matrix
    }

def wasserstein_distance(real_data: np.array, synthetic_data: np.array, columns):
    num_features = real_data.shape[2]
    real_eventlog = real_data.reshape(-1, num_features)
    synthetic_eventlog = synthetic_data.reshape(-1, num_features)

    cost_matrix = ot.dist(real_eventlog, synthetic_eventlog)
    distance = ot.emd2([], [], cost_matrix)
    return distance

def wasserstein_distance_timesteps(real_data: np.array, synthetic_data: np.array, columns):
    # Compute wasserstein distances between real and synthetic time-steps
    sequence_length = real_data.shape[1]
    distances = {}

    for timestep in range(sequence_length):
        timestep_real = real_data[:, timestep, :]
        timestep_syn = synthetic_data[:, timestep, :]

        cost_matrix = ot.dist(timestep_real, timestep_syn)
        distance = ot.emd2([], [], cost_matrix) 

        distances[f"timestep {timestep}"] = distance

    average_distance = np.mean(list(distances.values()))

    return {
        "average_distance": average_distance,
        "distances": distances
    }

def differences_timestep_distances(real_data: np.array, synthetic_data: np.array, columns):
    # Calculate distance matrix of real data, then of synthetic data, take difference
    timestep_distance_matrix = np.abs(distances_timesteps(real_data) - distances_timesteps(synthetic_data))

    # Calculate magnitude of distance matrix
    frobenius_norm = np.linalg.norm(timestep_distance_matrix, 'fro')

    # TODO: Visualise matrix

    return {
        "Magnitude of difference matrix": frobenius_norm,
        "differences in distances between timesteps": timestep_distance_matrix
    }

def distances_timesteps(data: np.array):
    # This function calculates the wasserstein distance matrix between time steps

    num_timesteps = data.shape[1]
    distance_matrix = np.zeros((num_timesteps, num_timesteps))

    # Loop through timesteps
    for t1 in range(num_timesteps):
        for t2 in range(t1, num_timesteps):

            # Don't calculate distance for similar timesteps
            if t1 == t2: continue

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

def differences_timestep_correlations(real_data: np.array, synthetic_data: np.array, columns):
    # This method calculates the differences in real and synthetic correlations between timesteps 

    num_features = real_data.shape[2]

    # keep track of differences in synthetic and real timestep correlations per variable
    magnitudes = {}
    diffs_timestep_corr = {}

    # loop through variables to calculate differences in timestep correlations
    for feature_idx in range(num_features):
        # Get events of feature (sequences, events)
        real_feature_events = real_data[:, :, feature_idx]
        # print(real_feature_events)

        synthetic_feature_events = synthetic_data[:, :, feature_idx]

        diffs_correlations, frobenius_norm = diff_corr_matrix(real_feature_events, synthetic_feature_events)

        magnitudes[columns[feature_idx]] = frobenius_norm
        # Add difference correlations to dictionary
        diffs_timestep_corr[columns[feature_idx]] = diffs_correlations

    return {
        "Magnitudes": magnitudes,
        "Differences real and synthetic time-step correlations": diffs_timestep_corr
    }

def diff_corr_matrix(real_events: np.array, synthetic_events: np.array):
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

    return diff_matrix, frobenius_norm

similarity_methods = [
    stats,
    kolmogorov_smirnov,
    differences_variable_correlations,
    wasserstein_distance,
    wasserstein_distance_timesteps,
    differences_timestep_distances,
    differences_timestep_correlations
]

# TODO: add correlations between timesteps as measure of short/long term dependencies