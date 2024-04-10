from utils import parse_args, load_data
from model import select_model


if __name__ == "__main__":
    # Parse arguments, load data, select and train model
    args = parse_args()
    values, columns = load_data(args.dataset)
    model = select_model(values, args)
    model.train(args.epochs, args.lr)

    # Generate samples
    num_samples = values.shape[0]
    model.generate_data(num_samples, args.dataset, args.epochs)

    print("Complete!")