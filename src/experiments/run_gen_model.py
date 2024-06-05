import argparse
import torch
import os
from torch.utils.data import random_split

from src.data.data_loader import select_data
from src.models.models import select_gen_model, GenModel
from src.training.hyperparameters import select_hyperparams
from src.models.models import train_model
from src.paths import PATH_SYNDATA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--exp_id', type=int, required=True)
    args = parser.parse_args()

    # data loading
    dataset = select_data(args.dataset)
    train_size = int(args.train_split * len(dataset))
    train_data, test_data = random_split(dataset, [train_size, len(dataset) - train_size])

    # train models
    model_class = select_gen_model(args.model)
    shape = (train_size, train_data[0].shape[0], train_data[0].shape[1])
    hyperparams = select_hyperparams(args.dataset, args.model, shape)
    hyperparams["epochs"] = 50
    model: GenModel = model_class(**hyperparams)
    train_model(model, train_data)

    # generate data
    if args.num_samples is None:
        syndata = model.generate_data(train_size)
    else:
        syndata = model.generate_data(args.num_samples)

    # save data
    path = f"{PATH_SYNDATA}/{args.exp_id}/"
    suffix = f"{args.dataset}-{args.model}-{train_size}.pt"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(syndata, f"{path}{suffix}")

    from src.data.data_loader import load_syn_data

    data = load_syn_data(f"{path}{suffix}")
    print(data)