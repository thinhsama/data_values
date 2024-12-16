# datasets.py
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pylab as plt
import torch

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
