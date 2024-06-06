import argparse

from src.data.data_loader import load_syn_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_samples')
    parser.add_argument('--exp', type=int, help="what experiment number?")
    args = parser.parse_args()

    # Load synthetic dataset
    path = f"outputs/syndata/{args.exp}/{args.dataset}-{args.model}-{args.num_samples}.pt"
    dataset = load_syn_data(path)
    # Create evaluator
    # evaluate
    # save results