import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml
import torch
def ensure_directory_exists(directory):
    """Ensure the directory exists, create it if not."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_openml(data_id: int, is_classification=True):
    """Load OpenML datasets.

    A helper function to load OpenML datasets using their data ID.
    """
    dataset = fetch_openml(data_id=data_id, as_frame=False)
    category_list = list(dataset.get("categories", {}).keys())
    if len(category_list) > 0:
        category_indices = [
            dataset["feature_names"].index(x)
            for x in category_list
            if x in dataset["feature_names"]
        ]
        noncategory_indices = [
            i for i in range(len(dataset["feature_names"])) if i not in category_indices
        ]
        X, y = dataset["data"][:, noncategory_indices], dataset["target"]
    else:
        X, y = dataset["data"], dataset["target"]

    # Label transformation
    if is_classification is True:
        list_of_classes, y = np.unique(y, return_inverse=True)
    else:
        y = (y - np.mean(y)) / (np.std(y.astype(float)) + 1e-8)

    # Ensure X is float
    if not np.issubdtype(X.dtype, np.number):
        X = X.astype(float)

    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)  # Standardization
    return X, y
def download_diabetes():
    """Regression data set registered as ``"diabetes"``."""
    return ds.load_diabetes(return_X_y=True)
def download_linnerud():
    """Regression data set registered as ``"linnerud"``."""
    return ds.load_linnerud(return_X_y=True)
def download_creditcard():
    """Categorical data set registered as ``"creditcard"``."""
    return load_openml(data_id=42477)
def download_wave_energy():
    """Regression data set registered as ``"wave_energy"``."""
    return load_openml(data_id=44975, is_classification=False)
def download_lowbwt():
    """Regression data set registered as ``"lowbwt"``."""
    return load_openml(data_id=1193, is_classification=False)
def download_mv():
    """Regression data set registered as ``"mv"``."""
    return load_openml(data_id=344, is_classification=False)
def download_stock():
    """Regression data set registered as ``"stock"``."""
    return load_openml(data_id=1200, is_classification=False)
def download_echoMonths():
    """Regression data set registered as ``"echoMonths"``."""
    return load_openml(data_id=1199, is_classification=False)
def split_data_train_val_test(X, y, train_size: float, valid_size: float, random_state: int = 1):
    """
    Split the data into train, validation, and test sets.

    Parameters:
        X (np.ndarray): Features.
        y (np.ndarray): Target labels/values.
        train_size (float): Proportion of data for training (0-1).
        valid_size (float): Proportion of data for validation (0-1).
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, y_train: Training set.
        X_valid, y_valid: Validation set.
        X_test, y_test: Test set.
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    train_split = int(train_size * len(X))
    valid_split = int((train_size + valid_size) * len(X))
    train_indices = indices[:train_split]
    valid_indices = indices[train_split:valid_split]
    test_indices = indices[valid_split:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_valid, y_valid = X[valid_indices], y[valid_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    print("Train:", X_train.shape, "Validation:", X_valid.shape, "Test:", X_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test
# if __name__ == "__main__":
#     # Load the diabetes dataset
#     X, y = download_diabetes()
#     X_train, y_train, X_valid, y_valid, X_test, y_test = split_data_train_val_test(
#         X, y, train_size=0.7, valid_size=0.15
#     )
#     print("Data split successfully.")
#     # Standardize the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_valid = scaler.transform(X_valid)
#     X_test = scaler.transform(X_test)
#     print("Data standardized successfully.")
#     # Convert the data to PyTorch tensors
#     X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
#     X_valid, y_valid = torch.tensor(X_valid), torch.tensor(y_valid)
#     X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
#     print("Data converted to PyTorch tensors successfully.")
#     print("Example data point:", X_train[0], y_train[0])
#     print("Example validation data point:", X_valid[0], y_valid[0])
#     print("Example test data point:", X_test[0], y_test[0])
#     print("Data preparation completed.")