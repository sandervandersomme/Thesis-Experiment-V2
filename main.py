from utils import parse_args, load_data
from model import select_model


if __name__ == "__main__":
    args, settings = parse_args()
    data = load_data(args.dataset)
    model = select_model(args, data)
    model.train(args.epochs, args.lr)
    # print(model.generate_data(5, args.dataset))

    print("Complete!")