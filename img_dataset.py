# img_dataset.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

from embedding_img import ImageEmbedder
def download_cifar10(data_dir: str = './data', train: bool = True) -> datasets.CIFAR10:
    """
    Download and return the CIFAR-10 dataset.

    Parameters:
        data_dir (str): Directory to store the dataset.
        train (bool): Whether to load the training set or test set.

    Returns:
        datasets.CIFAR10: CIFAR-10 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    return dataset


def get_dataloaders(data_dir: str = './data', batch_size: int = 32, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for the CIFAR-10 dataset.

    Parameters:
        data_dir (str): Directory to store the dataset.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test dataloaders.
    """
    train_dataset = download_cifar10(data_dir=data_dir, train=True)
    test_dataset = download_cifar10(data_dir=data_dir, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=4)

    # Display the size of the dataset
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Iterate over a batch of data
    for images, labels in train_loader:
        print(f"Batch size: {images.size()}, Labels size: {labels.size()}")
        break  # Display only the first batch for demonstration
        # Initialize ImageEmbedder
    print("Initializing ImageEmbedder with ResNet-18...")
    embedder = ImageEmbedder(model_name="resnet18")

    # Generate embeddings for the training set
    print("Generating embeddings for the training set...")
    train_embeddings, train_labels = embedder.generate_embeddings(train_loader)
    print(f"Training embeddings shape: {train_embeddings.shape}")
    print(f"First 10 training labels: {train_labels[:10]}")

    # Generate embeddings for the test set
    print("Generating embeddings for the test set...")
    test_embeddings, test_labels = embedder.generate_embeddings(test_loader)
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"First 10 test labels: {test_labels[:10]}")