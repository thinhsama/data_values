# subset.py
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from typing import Optional, Tuple
from datasets import download_iris
from torch.utils.data import TensorDataset
class SubsetDataset(Dataset):
    """
    Custom dataset wrapper to extract a subset of a given dataset.

    Parameters:
        dataset (Dataset): PyTorch dataset.
        subset_size (int): Number of samples to include in the subset.
        random_state (Optional[int]): Random seed for reproducibility.
    """
    def __init__(self, dataset: Dataset, subset_size: int = 1000, random_state: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.subset_size = subset_size
        self.random_state = random_state
        self.subset_indices = self._get_subset_indices()

    def _get_subset_indices(self) -> torch.Tensor:
        """
        Randomly select a subset of indices from the dataset.

        Returns:
            torch.Tensor: Indices of the selected subset.
        """
        total_size = len(self.dataset)
        torch.manual_seed(self.random_state)
        indices = torch.randperm(total_size)[:self.subset_size]
        return indices

    def __len__(self) -> int:
        """
        Return the size of the subset.
        """
        return len(self.subset_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a data sample and its label by index.

        Parameters:
            index (int): Index of the sample in the subset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data sample and label.
        """
        original_index = self.subset_indices[index].item()
        return self.dataset[original_index]
def create_subset_loader(feature:torch.Tensor, target:torch.Tensor, subset_size=1000, batch_size=32, random_state=42):
    feature = torch.tensor(feature, dtype=torch.float32) if not isinstance(feature, torch.Tensor) else feature
    target = torch.tensor(target, dtype=torch.long) if not isinstance(target, torch.Tensor) else target
    full_dataset = TensorDataset(feature, target)
    subset_dataset = SubsetDataset(dataset=full_dataset, subset_size=subset_size, random_state=random_state)
    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

# if __name__ == "__main__":
#     feature, target = download_iris()
#     subset_loader = create_subset_loader(feature, target, subset_size=20, batch_size=5, random_state=42)
#     # print(full_dataset[0])
#     # # Create a subset of the dataset
#     # subset_size = 10
#     # random_state = 42
#     # subset_dataset = SubsetDataset(dataset=full_dataset, subset_size=subset_size, random_state=random_state)

#     # # Wrap subset in a DataLoader
#     # batch_size = 6
#     # subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
#     # Print out some information
#     # print(f"Full dataset size: {len(full_dataset)}")
#     # print(f"Subset size: {len(subset_dataset)}")
#     # Iterate through the subset
#     for batch_idx, (data, labels) in enumerate(subset_loader):
#         print(f"Batch {batch_idx + 1}: Data shape: {data.shape}, Labels shape: {labels.shape}")
#         break  # Just one batch for demonstration
