import pandas as pd
import torch

def evaluation(models, datasets):
    results = pd.Dataframe()

    for model in models:
        for dataset in datasets:
            