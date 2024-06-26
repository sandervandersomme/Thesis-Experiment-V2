import torch
import ot
import numpy as np
from typing import List
from scipy.stats import ks_2samp, skew, kurtosis
import pandas as pd

# from src.eval.visualise import visualise_varcor_similarities, visualise_distributions

def similarity_of_statistics(real_data: torch.Tensor, syndata: torch.Tensor, columns: List[str]):
    # Convert tensors to correct shape
    real_data = real_data.cpu().numpy().reshape(-1, real_data.shape[2])
    syndata = syndata.cpu().numpy().reshape(-1, syndata.shape[2])

    metrics = ["mean", "std", "median", "var"] # Removed skewness and kurtosis
    methods = [np.mean, np.std, np.median, np.var]
    
    # Initialize a DataFrame to store similarities
    stat_similarities_per_variable_df = pd.DataFrame(index=columns, columns=metrics)
    
    # Loop through statistic methods
    for method, metric in zip(methods, metrics):

        # Calculate differences and average difference in real and synthetic variable statistics
        real_statistics = method(real_data, axis=0)
        syn_statistics = method(syndata, axis=0)
        stat_similarities_per_variable = 1 - np.abs((real_statistics - syn_statistics) / np.maximum(np.abs(real_statistics), np.abs(syn_statistics)))
        
        # Update the DataFrame
        stat_similarities_per_variable_df[metric] = stat_similarities_per_variable

    return stat_similarities_per_variable_df

def similarity_of_correlations(real_data: torch.Tensor, syndata: torch.Tensor):
    # Calculates the difference between correlation matrices of real and synthetic data 
    # (i.e. how do correlations between variable pair differ between real and synthetic data)

    # Flatten the sequences into tabular format: (Events, number of features)
    num_features = real_data.size(2)
    real_eventlog = real_data.cpu().numpy().reshape(-1, num_features)
    synthetic_eventlog = syndata.cpu().numpy().reshape(-1, num_features)
    similarity_matrix, similarity_score = similarity_correlation_matrix(real_eventlog, synthetic_eventlog)

    return similarity_score, similarity_matrix

def wasserstein_distance(real_data: torch.Tensor, syndata: torch.Tensor, columns):
    """
    Calculates the wasserstein distance between real and synthetic eventlogs
    """
    num_features = real_data.shape[2]
    real_eventlog = real_data.cpu().numpy().reshape(-1, num_features)
    synthetic_eventlog = syndata.cpu().numpy().reshape(-1, num_features)

    cost_matrix = ot.dist(real_eventlog, synthetic_eventlog)
    distance = ot.emd2([], [], cost_matrix, numItermax=200000)

    return distance

def similarity_correlation_matrix(real_events: np.ndarray, synthetic_events: torch.Tensor):
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
    normalised_frobenius_norm = frobenius_norm / similarity_matrix.shape[0]

    return similarity_matrix, normalised_frobenius_norm

if __name__ == "__main__":
    shape = (200, 5, 20)
    realdata = torch.rand(shape)
    syn = torch.rand(shape)
    columns = [str(x) for x in range(realdata.shape[2])]

    matrix, avgs = similarity_of_statistics(realdata, syn, columns)
    print(avgs)
    print(**dict(avgs))