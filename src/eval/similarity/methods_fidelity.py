import torch
import ot
import numpy as np
from typing import List
from scipy.stats import ks_2samp, skew, kurtosis

from src.eval.visualise import visualise_varcor_similarities, visualise_distributions


def stats(test_data: torch.Tensor, syndata: torch.Tensor, columns: List[str]):
    # Convert tensors to correct shape
    test_data = test_data.numpy().reshape(-1, test_data.shape[2])
    syndata = syndata.numpy().reshape(-1, syndata.shape[2])

    scores = {}
    var_diffs = {}
    
    # Loop through statistic methods
    for method in [np.mean, np.std, np.median, np.var, skew, kurtosis]:

        # Calculate differences and average difference in real and synthetic variable statistics
        differences = np.abs(method(test_data, axis=0) - method(syndata, axis=0))
        average_diff = np.mean(differences)

        scores.update({
            f"Average {method.__name__}": average_diff
        })
            
        var_diffs[method.__name__] = dict(zip(columns, differences))

    # scores.update({
    #     "differences per variable": var_diffs
    # })

    return scores

def kolmogorov_smirnov(test_data: torch.Tensor, syndata: torch.Tensor, columns: List[str]):
    "Use kolmogorov smirnov to calculate distances between variable distributions"

    # Track ks-sample test statistics and p_values
    statistics = {}
    p_values = {}

    # Loop through variable distributions
    num_features = test_data.shape[2]
    for feature_idx in range(num_features):
        real_feature = test_data[:, :, feature_idx].flatten()
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

def avg_diff_correlations(test_data: torch.Tensor, syndata: torch.Tensor, graph_path):
    # Calculates the difference between correlation matrices of real and synthetic data 
    # (i.e. how do correlations between variable pair differ between real and synthetic data)

    # Flatten the sequences into tabular format: (Events, number of features)
    num_features = test_data.size(2)
    real_eventlog = test_data.numpy().reshape(-1, num_features)
    synthetic_eventlog = syndata.numpy().reshape(-1, num_features)
    diff_matrix, frob_norm = similarity_correlation_matrix(real_eventlog, synthetic_eventlog)

    visualise_varcor_similarities(diff_matrix, f"{graph_path}sim_varcors.png")

    return {
        "Magnitude of difference in variable correlations": frob_norm
    }

def wasserstein_distance(test_data: torch.Tensor, syndata: torch.Tensor, columns, graph_path: str):
    """
    Calculates the wasserstein distance between real and synthetic eventlogs
    """
    num_features = test_data.shape[2]
    real_eventlog = test_data.numpy().reshape(-1, num_features)
    synthetic_eventlog = syndata.numpy().reshape(-1, num_features)

    cost_matrix = ot.dist(real_eventlog, synthetic_eventlog)
    distance = ot.emd2([], [], cost_matrix)

    visualise_distributions(real_eventlog, synthetic_eventlog, columns, f"{graph_path}distributions/")

    return {"Wasserstein distance between real and synthetic eventlogs": distance}

def similarity_correlation_matrix(real_events: torch.Tensor, synthetic_events: torch.Tensor):
    # Calculate the differences between real and synthetic correlation matrices
    corr_matrix_real = np.corrcoef(real_events, rowvar=False)
    corr_matrix_syn = np.corrcoef(synthetic_events, rowvar=False)

    # Fix issue with nan-values: non-correlations
    corr_matrix_real = np.nan_to_num(corr_matrix_real)
    corr_matrix_syn = np.nan_to_num(corr_matrix_syn)

    # calculate the difference between the correlation matrices
    diff_matrix = np.abs(corr_matrix_real - corr_matrix_syn)
    
    # Convert into similarity matrix by normalising and inversing
    similarity_matrix = 1 - (diff_matrix/2)

    # The frobenius norm calculates the size or magnitude of a matrix (in this case, the magnitude of the difference matrix)
    frobenius_norm = np.linalg.norm(similarity_matrix, 'fro')

    return similarity_matrix, frobenius_norm