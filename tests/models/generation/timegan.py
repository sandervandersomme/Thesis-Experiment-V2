from src.data.data_loader import select_data
from src.models.generative.timegan import TimeGAN, train_TimeGAN, TimeGAN_params

import torch

if __name__ == "__main__":
    # Load data
    dataset = select_data("cf_classification")
    numseq, numev, numfeat = dataset.sequences.shape

    # Load parameters
    TimeGAN_params.update({
        "num_sequences": numseq,
        "num_events": numev,
        "num_features": numfeat
    })

    # Train model
    model = TimeGAN(**TimeGAN_params)
    model.output_path = f"outputs/testing/genmodels/"
    train_TimeGAN(model, dataset, dataset, log_dir=f"runs/testing/{model.__NAME__}/")

    # Generate and save data
    syndata = model.generate_data(numseq)
    torch.save(syndata, f"{model.output_path}/{model.__NAME__}/syndata")
    print("Saving data..")
