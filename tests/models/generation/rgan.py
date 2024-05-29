from src.data.data_loader import select_data
from src.models.generative.rgan import RGAN, train_RGAN
from src.models.generative.gen_model import gen_params

import torch

if __name__ == "__main__":
    # Load data
    dataset = select_data("cf")
    numseq, numev, numfeat = dataset.sequences.shape

    # Load parameters
    gen_params.update({
        "num_sequences": numseq,
        "num_events": numev,
        "num_features": numfeat
    })

    # Train model
    model = RGAN(**gen_params)
    model.output_path = f"outputs/testing/genmodels/"
    
    train_RGAN(model, dataset, log_dir=f"runs/testing/{model.__NAME__}/")

    # Generate and save data
    syndata = model.generate_data(numseq)
    torch.save(syndata, f"{model.output_path}/{model.__NAME__}/syndata")
    print("Saving data..")

    