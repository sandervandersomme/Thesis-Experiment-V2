import torch
from scipy.stats import ks_2samp, gaussian_kde
from scipy.spatial.distance import jensenshannon
from typing import List
import matplotlib.pyplot as plt
import numpy as np

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