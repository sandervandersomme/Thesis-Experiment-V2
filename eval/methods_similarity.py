import torch
import numpy as np

from scipy.stats import ks_2samp
import ot


from typing import List

def difference_mean(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str]):
    # calculate differences in mean per variable, then average difference across variables
    mean_real = torch.mean(real_data, dim=1)
    mean_syn = torch.mean(synthetic_data, dim=1)
    diff_mean = torch.abs(mean_real - mean_syn)
    average_mean = torch.mean(diff_mean)

    return {
        "Average difference" : diff_mean,
        "differences in averages per variable" : dict(zip(columns, diff_mean))
    }

def difference_std(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str]):
    # calculate differences in std per variable, then average difference across variables
    std_real = torch.std(real_data, dim=1)
    std_syn = torch.std(synthetic_data, dim=1)
    diff_std = torch.abs(std_real - std_syn)
    average_std = torch.std(diff_std)

    return {
        "Average std" : diff_std,
        "STD differences per variable" : dict(zip(columns, diff_std))
    }

def difference_median(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str]):
    # calculate differences in median per variable, then average difference across variables
    median_real = torch.median(real_data, dim=1).values
    median_syn = torch.median(synthetic_data, dim=1).values
    diff_median = torch.abs(median_real - median_syn)
    average_median = torch.median(diff_median)

    return {
        "Average difference" : diff_median,
        "Column differences" : dict(zip(columns, average_median))
    }

def difference_skewness(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str]):
    # calculate differences in median per variable, then average difference across variables
    skewness_real = torch.median(real_data, dim=1).values
    skewness_syn = torch.median(synthetic_data, dim=1).values
    diff_skewness = torch.abs(skewness_real - skewness_syn)
    average_skewness = torch.median(diff_skewness)

    return {
        "Average difference" : diff_skewness,
        "Column differences" : dict(zip(columns, average_skewness))
    }

def difference_kurtosis(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str]):
    # calculate differences in median per variable, then average difference across variables
    kurtosis_real = torch.median(real_data, dim=1).values
    kurtosis_syn = torch.median(synthetic_data, dim=1).values
    diff_kurtosis = torch.abs(kurtosis_real - kurtosis_syn)
    average_kurtosis = torch.median(diff_kurtosis)

    return {
        "Average difference" : diff_kurtosis,
        "Column differences" : dict(zip(columns, average_kurtosis))
    }

def difference_statistics(real_data: torch.Tensor, synthetic_data: torch.Tensor):
    # Convert tensors to correct shape
    real_data = real_data.reshape(-1, real_data.shape[2])
    synthetic_data = synthetic_data.reshape(-1, synthetic_data.shape[2])

    return {
        "mean" : difference_mean(real_data, synthetic_data),
        "std" : difference_std(real_data, synthetic_data),
        "median" : difference_median(real_data, synthetic_data),
        "skewness" : difference_skewness(real_data, synthetic_data),
        "kurtosis" : difference_kurtosis(real_data, synthetic_data)
    }

def kolmogorov_smirnov(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str]):
    real_data = real_data.numpy()
    synthetic_data = synthetic_data.numpy()

    # Track ks-sample test statistics and p_values
    statistics = {}
    p_values = {}

    # Extract number of features
    num_features = real_data.shape[2]
    assert num_features == len(columns)

    # Loop through variable distributions
    for feature_idx in range(num_features):
        real_feature = extract_feature_distribution(real_data, feature_idx)
        synthetic_feature = extract_feature_distribution(synthetic_data, feature_idx)

        # Calculate the Kolomgorov-Smirnov two sample test
        distance, p_value = ks_2samp(real_feature, synthetic_feature)

        # Track results
        statistics[columns[feature_idx]] = distance
        p_values[columns[feature_idx]] = p_value

    return {
        "statistics" : dict(zip(columns, statistics)),
        "p_values" : dict(zip(columns, p_values))
    }

def extract_feature_distribution(data: np.array, feature_idx: int):
    # Extract a single feature
    return data[:, :, feature_idx].flatten() 

def pearson_correlation_difference(real_data: np.array, synthetic_data: np.array, columns):
    real_data = real_data.numpy()
    synthetic_data = synthetic_data.numpy()

    # Flatten the sequences into tabular format: (Events, number of features)
    real_eventlog = extract_eventlog(real_data)
    synthetic_eventlog = extract_eventlog(synthetic_data)

    # Calculate the Pearson correlation matrices of the real and synthetic datasets
    corr_matrix_real = np.corrcoef(real_eventlog, rowvar=False)
    corr_matrix_syn = np.corrcoef(synthetic_eventlog, rowvar=False)

    # calculate the difference between the correlation matrices
    diff_matrix = torch.abs(corr_matrix_real - corr_matrix_syn)

    # The frobenius norm calculates the size or magnitude of a matrix (in this case, the magnitude of the difference matrix)
    frobenius_norm = np.linalg.norm(diff_matrix, 'fro')

    return frobenius_norm

def extract_eventlog(data):
    # Flatten the sequences into tabular format: (Events, number of features)
    return data.reshape(-1, data.shape[2])

def wasserstein_distance(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns):
    real_eventlog = extract_eventlog(real_data.numpy()) 
    synthetic_eventlog = extract_eventlog(synthetic_data.numpy())

    cost_matrix = ot.dist(real_eventlog, synthetic_eventlog)
    distance = ot.emd2([], [], cost_matrix)
    return distance

def wasserstein_distance_timesteps(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns):
    sequence_length = real_data.size(1)
    timestep_distances = {}

    for timestep in range(sequence_length):
        timestep_real = extract_timestep_distribution(real_data.numpy(), timestep) 
        timestep_syn = extract_timestep_distribution(synthetic_data.numpy(), timestep)
            
        cost_matrix = ot.dist(timestep_real, timestep_syn)
        distance = ot.emd2([], [], cost_matrix) 

        timestep[f"timestep {timestep}"] = distance

    return timestep_distances

def extract_timestep_distribution(data: np.array, timestep: int):
    # Flatten the sequences into tabular format: (All sequences, one timestep, all features)
    return data[:, timestep, :]

similarity_methods = [
    difference_statistics,
    kolmogorov_smirnov,
    pearson_correlation_difference,
    wasserstein_distance,
    wasserstein_distance_timesteps
]