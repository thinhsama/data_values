# main.py
import argparse
import torch
import numpy as np
from knn_shapley import KNNEvaluator
from datasets import download_iris, split_data_train_val_test

def main(args):
    # Load dataset
    if args.dataset == 'iris':
        print("Loading Iris dataset...")
        X, y = download_iris()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Split data into train, validation, and test sets
    print("Splitting data...")
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data_train_val_test(
        X, y, train_size=args.train_size, valid_size=args.valid_size, random_state=args.random_state
    )

    # Convert data to PyTorch tensors
    x_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.long)

    # Initialize KNNEvaluator
    print(f"Initializing KNNEvaluator with k_neighbors={args.k_neighbors} and batch_size={args.batch_size}...")
    knn_evaluator = KNNEvaluator(k_neighbors=args.k_neighbors, batch_size=args.batch_size, random_state=args.random_state)

    # Evaluate data values
    print("Evaluating data values...")
    data_values = knn_evaluator.evaluate_data_values(x_train, y_train, x_valid, y_valid)

    # Print results
    print("\nTop 10 data values:")
    print(data_values[:10])

    if args.save_path:
        np.save(args.save_path, data_values)
        print(f"\nData values saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Shapley Value Computation")
    parser.add_argument("--dataset", type=str, default="iris", help="Dataset to use (default: iris)")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of training data (default: 0.7)")
    parser.add_argument("--valid_size", type=float, default=0.2, help="Proportion of validation data (default: 0.2)")
    parser.add_argument("--k_neighbors", type=int, default=10, help="Number of neighbors for KNN (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for distance computation (default: 32)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--save_path", type=str, help="Path to save computed data values (optional)")
    args = parser.parse_args()

    main(args)
# python main.py --dataset iris --train_size 0.7 --valid_size 0.2 --k_neighbors 15 --batch_size 16 --random_state 123 --save_path data_values.npy
#python -m venv env
#source env/bin/activate   # Với Linux/macOS
#env\Scripts\activate      # Với Windows