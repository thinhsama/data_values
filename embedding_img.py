# embedding_img.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from typing import List, Tuple
import numpy as np


class ImageEmbedder:
    """
    Class for generating embeddings from images using a pre-trained model.

    Parameters:
        model_name (str): Name of the pre-trained model to use (default: 'resnet18').
        device (str): Device to run the model on (default: 'cuda' if available, else 'cpu').
    """
    def __init__(self, model_name: str = "resnet18", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def _load_model(self, model_name: str) -> nn.Module:
        """
        Load a pre-trained model and remove the final classification layer.

        Parameters:
            model_name (str): Name of the pre-trained model.

        Returns:
            nn.Module: Pre-trained model without the classification layer.
        """
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model

    def generate_embeddings(self, dataloader: DataLoader) -> Tuple[np.ndarray, List[int]]:
        """
        Generate embeddings for a dataset.

        Parameters:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            Tuple[np.ndarray, List[int]]: Array of embeddings and corresponding labels.
        """
        embeddings = []
        labels = []
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                features = self.model(images)  # Extract features
                features = features.view(features.size(0), -1)  # Flatten
                embeddings.append(features.cpu().numpy())
                labels.extend(targets.numpy())

        embeddings = np.vstack(embeddings)  # Combine all embeddings
        return embeddings, labels


# if __name__ == "__main__":
#     from img_dataset import get_dataloaders

#     # Parameters
#     batch_size = 64
#     num_workers = 4

#     # Load CIFAR-10 dataset
#     print("Loading CIFAR-10 dataset...")
#     train_loader, test_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

#     # Initialize ImageEmbedder
#     print("Initializing ImageEmbedder with ResNet-18...")
#     embedder = ImageEmbedder(model_name="resnet18")

#     # Generate embeddings for the training set
#     print("Generating embeddings for the training set...")
#     train_embeddings, train_labels = embedder.generate_embeddings(train_loader)
#     print(f"Training embeddings shape: {train_embeddings.shape}")
#     print(f"First 10 training labels: {train_labels[:10]}")

#     # Generate embeddings for the test set
#     print("Generating embeddings for the test set...")
#     test_embeddings, test_labels = embedder.generate_embeddings(test_loader)
#     print(f"Test embeddings shape: {test_embeddings.shape}")
#     print(f"First 10 test labels: {test_labels[:10]}")
