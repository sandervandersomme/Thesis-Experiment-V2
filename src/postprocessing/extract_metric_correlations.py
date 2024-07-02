import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    args = parser.parse_args()

    # path = os.path.join("outputs/", f"{args.dir}/results/full/all_scores.csv")
    path = os.path.join("outputs/", f"{args.dir}/results/avgs/all_avg_scores.csv")

    df = pd.read_csv(path, index_col=0).T
    corr_matrix = df.corr()

    plt.figure(figsize=(12,8), dpi = 100)
    mask = np.triu(np.ones_like(corr_matrix))
    sns.heatmap(corr_matrix, annot = True, fmt = '.2f', center = 0, vmin=-1, vmax=1, cmap="RdBu", mask=mask, annot_kws={"fontsize":7})
    plt.title("Correlations between metrics")
    plt.xticks(fontsize=7, rotation=60, horizontalalignment='right')
    plt.yticks(fontsize=7)
    plt.show()

