from src.data.cf_classification import CF_Classification
from src.data.cf import CF
import torch
import pandas as pd

# TODO: Add random split

model = "RWGAN"

def save_sequences_and_labels(model: str):
    # Load synthetic data by model 
    syn_path = f"outputs/testing/{model}/syn.pt"
    syndata = torch.load(syn_path)

    # Create labels within synthetic dataset
    cf_syn = CF_Classification(syndata, cf.columns)

    # Store sequences
    df = pd.DataFrame(cf_syn.sequences.numpy().reshape(-1, cf_syn.sequences.shape[2]), columns=cf_syn.columns)
    df.to_csv(f"outputs/testing/syndata/{model}/sequences.csv", index=False)

    # Save labels
    df = pd.DataFrame(cf_syn.labels.bool().numpy(), columns=["improved?"])
    df.to_csv(f"outputs/testing/syndata/{model}/labels.csv", index=False)

    


if __name__ == "__main__":
    # load real data
    cf = CF()
    cf_class = CF_Classification(cf.sequences, cf.columns)
    df = pd.DataFrame(cf_class.labels.bool().numpy(), columns=["improved?"])
    df.to_csv("outputs/testing/syndata/labelsreal.csv", index=False)

    save_sequences_and_labels("RGAN")
    save_sequences_and_labels("RWGAN")
    save_sequences_and_labels("TimeGAN")



