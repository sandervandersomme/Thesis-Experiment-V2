import torch
from scipy.stats import ks_2samp, gaussian_kde
from scipy.spatial.distance import jensenshannon
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import ot

def difference_statistics(real_data: torch.Tensor, synthetic_data: torch.Tensor):
    # Convert tensors to correct shape
    real_data = real_data.reshape(-1, real_data.shape[2])
    synthetic_data = synthetic_data.reshape(-1, synthetic_data.shape[2])

    # calculate differences in mean per variable, then average difference across variables
    mean_real = torch.mean(real_data, dim=1)
    mean_syn = torch.mean(synthetic_data, dim=1)
    diff_mean = torch.abs(mean_real - mean_syn)
    average_mean = torch.mean(diff_mean)

    # calculate differences in std per variable, then average difference across variables
    std_real = torch.std(real_data, dim=1)
    std_syn = torch.std(synthetic_data, dim=1)
    diff_std = torch.abs(std_real - std_syn)
    average_std = torch.std(diff_std)

    # calculate differences in median per variable, then average difference across variables
    median_real = torch.median(real_data, dim=1).values
    median_syn = torch.median(synthetic_data, dim=1).values
    diff_median = torch.abs(median_real - median_syn)
    average_median = torch.median(diff_median)

    return average_mean, average_std, average_median

def univariate_distances(real_data: torch.Tensor, synthetic_data: torch.Tensor, columns: List[str], path: str):
    # Convert tensors to numpy
    real_data = real_data.numpy()
    synthetic_data = synthetic_data.numpy()

    # Calculate Kolmogorov-Smirnov statistics and p-values
    ks_statistics, p_values = kolmogorov_smirnov(real_data, synthetic_data)

    # Visualise results
    visualise_ks(ks_statistics, columns, "KS statistics",  f"output/eval/similarity/ks_stats_{path}")
    visualise_ks(p_values, columns, "P-values",  f"output/eval/similarity/ks_pvalues_{path}")

def bivariate_distances(real_data: torch.Tensor, synthetic_data: torch.Tensor):
    pwcd = pearson_correlation_difference(real_data, synthetic_data)

    # TODO: VISUALISE RESULTS
    return pwcd

def multivariate_distances(real_data: torch.Tensor, synthetic_data: torch.Tensor, path: str):
    events_real, events_syn = extract_eventlog(real_data.numpy(), synthetic_data.numpy())

    # Calculate difference in joint distribution across time-steps (ignore time-component)
    distance = wasserstein_distance(events_real, events_syn)

    # Calculate differences in joint distributions per time-step
    sequence_length = real_data.size(1)
    distances = []
    for timestep in range(sequence_length):
        timestep_real, timestep_syn = extract_timestep_distribution(real_data, synthetic_data, timestep)
        distances.append(wasserstein_distance(timestep_real, timestep_syn))

    visualise_timesteps(distances, list(range(sequence_length)), f"output/eval/similarity/wasserstein_timesteps_{path}")

    return distance, distances

def kolmogorov_smirnov(real_data: np.array, synthetic_data: np.array):
    # Track ks-sample test statistics and p_values
    statistics = []
    p_values = []
    num_features = real_data.shape[2]

    # Loop through variable distributions
    for feature_idx in range(num_features):
        real_feature_dis, synthetic_feature_dis = extract_feature_distribution(real_data, synthetic_data, feature_idx)

        # Calculate the Kolomgorov-Smirnov two sample test
        distance, p_value = ks_2samp(real_feature_dis, synthetic_feature_dis)

        # Track results
        statistics.append(distance)
        p_values.append(p_value)
    return statistics, p_values

def pearson_correlation_difference(real_data: np.array, synthetic_data: np.array):
    # Flatten the sequences into tabular format: (Events, number of features)
    real_data, synthetic_data = extract_eventlog(real_data, synthetic_data)

    # Calculate the Pearson correlation matrices of the real and synthetic datasets
    corr_matrix_real = np.corrcoef(real_data, rowvar=False)
    corr_matrix_syn = np.corrcoef(synthetic_data, rowvar=False)

    # calculate the difference between the correlation matrices
    diff_matrix = torch.abs(corr_matrix_real - corr_matrix_syn)

    # The frobenius norm calculates the size or magnitude of a matrix (in this case, the magnitude of the difference matrix)
    frobenius_norm = np.linalg.norm(diff_matrix, 'fro')

    return frobenius_norm

def wasserstein_distance(real_data: np.array, syn_data: np.array):
    cost_matrix = ot.dist(real_data, syn_data)
    distance = ot.emd2([], [], cost_matrix)
    return distance

def extract_eventlog(real_data, syn_data):
    # Flatten the sequences into tabular format: (Events, number of features)
    real_data = real_data.reshape(-1, real_data.shape[2])
    syn_data = syn_data.reshape(-1, syn_data.shape[2])

    return real_data, syn_data

def extract_feature_distribution(real_data, syn_data, feature_idx):
    # Flatten the sequences into tabular format
    feature_real = real_data[:, :, feature_idx].flatten() 
    feature_syn = syn_data[:, :, feature_idx].flatten()

    return feature_real, feature_syn

def extract_timestep_distribution(real_data, syn_data, timestep):
    # Flatten the sequences into tabular format: (All sequences, all time-steps, single feature)
    timestep_real = real_data[:, timestep, :]
    timestep_synthetic = syn_data[:, timestep, :]

    return timestep_real, timestep_synthetic    

def visualise_ks(data: List[float], variables: List[str], y_label: str, path: str):
    # Creating the plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.bar(variables, data, color='skyblue')  # Creates a bar plot

    # Add labels
    plt.xlabel('Variables')  # X-axis label
    plt.ylabel('y_label')  # Y-axis label
    plt.title(f'{y_label} per Variable')  # Plot title
    plt.xticks(rotation=45)  # Rotate variable names for better readability

    # Add the values above each bar
    for i, statistic in enumerate(data):
        plt.text(i, statistic + 0.01, f'{statistic:.2f}', ha='center')

    # Display the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(f"{path}.png")

def visualise_timesteps(data: List[float], timesteps: List[int], path: str):
    # Creating the plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.bar(timesteps, data, color='skyblue')  # Creates a bar plot

    # Add labels
    plt.xlabel('Timesteps')  # X-axis label
    plt.ylabel('Wasserstein distance')  # Y-axis label
    plt.title(f'Wasserstein distance per Timestep')  # Plot title
    plt.xticks(rotation=45)  # Rotate variable names for better readability

    # Add the distance values above each bar
    for i, distance in enumerate(data):
        plt.text(i, distance + 0.01, f'{distance:.2f}', ha='center')

    # Display the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(f"{path}.png")