"""
NLP and Tabular Datasets

Consolidated dataset loading functions, including synthetic and real-world datasets.
"""
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

def download_iris():
    """
    Load the Iris dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return ds.load_iris(return_X_y=True)

def gaussian_classifier(n: int = 10000, input_dim: int = 10):
    """
    Generate synthetic data for a Gaussian classifier.

    Parameters:
        n (int): Number of samples.
        input_dim (int): Number of input dimensions.

    Returns:
        covar (np.ndarray): Covariates (features).
        y (np.ndarray): Binary target labels.
    """
    covar = np.random.normal(size=(n, input_dim))
    beta_true = np.random.normal(size=input_dim).reshape(input_dim, 1)
    p_true = np.exp(covar.dot(beta_true)) / (1.0 + np.exp(covar.dot(beta_true)))
    y = np.random.binomial(1, p_true).reshape(-1)
    return covar, y

def download_digits():
    """
    Load the Digits dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return ds.load_digits(return_X_y=True)

def download_election(cache_dir: str, force_download: bool = False):
    """
    Load the Election Results dataset.

    Parameters:
        cache_dir (str): Directory to cache the dataset.
        force_download (bool): Force re-download of the dataset.

    Returns:
        X, y: Features and target labels.
    """
    ensure_directory_exists(cache_dir)
    url = "https://dataverse.harvard.edu/api/access/datafile/4299753?gbrecs=false"
    file_path = Path(cache_dir) / "election.tab"

    if not file_path.exists() or force_download:
        df = pd.read_csv(url, delimiter="\t")
        df.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path, delimiter="\t")

    drop_col = ["notes", "party_detailed", "candidate", "version", "state_po", "writein", "office"]
    df = df.drop(drop_col, axis=1)
    df = pd.get_dummies(df, columns=["state"])
    X = df.drop("party_simplified", axis=1).astype("float").values
    y = df["party_simplified"].astype("category").cat.codes.values
    return X, y

def download_linnerud():
    """
    Load the Linnerud dataset (regression problem).
    Returns:
        X, y: Features and target values.
    """
    return ds.load_linnerud(return_X_y=True)

def download_2dplanes():
    """
    Load the 2D Planes dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return load_openml(data_id=727)

def download_pol():
    """
    Load the POL dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return load_openml(data_id=722)

def download_fried():
    """
    Load the Fried dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return load_openml(data_id=901)

def download_nomao():
    """
    Load the Nomao dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return load_openml(data_id=1486)

def download_creditcard():
    """
    Load the Credit Card Fraud dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return load_openml(data_id=42477)

def download_wave_energy():
    """
    Load the Wave Energy dataset (regression problem).
    Returns:
        X, y: Features and target values.
    """
    return load_openml(data_id=44975, is_classification=False)

def download_stock():
    """
    Load the Stock dataset (regression problem).
    Returns:
        X, y: Features and target values.
    """
    return load_openml(data_id=1200, is_classification=False)

def download_breast_cancer():
    """
    Load the Breast Cancer dataset (classification problem).
    Returns:
        X, y: Features and target labels.
    """
    return ds.load_breast_cancer(return_X_y=True)

def download_diabetes():
    """
    Load the Diabetes dataset (regression problem).
    Returns:
        X, y: Features and target values.
    """
    return ds.load_diabetes(return_X_y=True)

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


def create_unbalanced_dataset(x_embeddings, y_labels, target_class=6, target_samples=5000, other_samples=2500):
    unique_classes = torch.unique(y_labels).cpu().numpy()
    
    x_unbalanced = []
    y_unbalanced = []
    
    for cls in unique_classes:
        indices = torch.where(y_labels == cls)[0]
        if cls == target_class:
            selected_indices = np.random.choice(indices.cpu(), target_samples, replace=True)
        else:
            selected_indices = np.random.choice(indices.cpu(), other_samples, replace=False)
        
        x_unbalanced.append(x_embeddings[selected_indices])
        y_unbalanced.append(y_labels[selected_indices])
    
    x_unbalanced = torch.cat(x_unbalanced, dim=0)
    y_unbalanced = torch.cat(y_unbalanced, dim=0)
    
    # Shuffle dữ liệu để đảm bảo tính ngẫu nhiên
    shuffled_indices = torch.randperm(len(y_unbalanced))
    x_unbalanced = x_unbalanced[shuffled_indices]
    y_unbalanced = y_unbalanced[shuffled_indices]
    
    return x_unbalanced, y_unbalanced

def check_dataset_balance(y_labels, dataset_name="Original"):
    class_counts = torch.bincount(y_labels)
    total_samples = len(y_labels)
    print(f"\n{dataset_name} Dataset Class Distribution:")
    for cls, count in enumerate(class_counts):
        print(f"Class {cls}: {count} samples ({(count / total_samples) * 100:.2f}%)")

# if __name__ == "__main__":
#     # Test Iris dataset
#     print("\nIris Dataset:")
#     X, y = download_iris()
#     print("Iris Dataset Shape:", X.shape, y.shape)

#     # Test Gaussian Classifier
#     print("\nGaussian Classifier:")
#     X_gaussian, y_gaussian = gaussian_classifier(n=10, input_dim=5)
#     print("Gaussian Classifier Shape:", X_gaussian.shape, y_gaussian.shape)

#     # Test Breast Cancer dataset
#     print("\nBreast Cancer Dataset:")
#     X_bc, y_bc = download_breast_cancer()
#     print("Breast Cancer Dataset Shape:", X_bc.shape, y_bc.shape)

#     # Test Diabetes dataset
#     print("\nDiabetes Dataset:")
#     X_diabetes, y_diabetes = download_diabetes()
#     print("Diabetes Dataset Shape:", X_diabetes.shape, y_diabetes.shape)

#     # Test splitting data
#     print("\nSplit Data (Iris Dataset):")
#     X_train, y_train, X_valid, y_valid, X_test, y_test = split_data_train_val_test(X, y, train_size=0.7, valid_size=0.2)
#     print("Train Set Shape:", X_train.shape, y_train.shape)
#     print("Validation Set Shape:", X_valid.shape, y_valid.shape)
#     print("Test Set Shape:", X_test.shape, y_test.shape)
