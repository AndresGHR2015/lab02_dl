import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CognitiveDataset(Dataset):
    """
    A custom PyTorch Dataset for loading tabular features and corresponding multilabel targets.
    Abstracts away the underlying pandas DataFrames or numpy arrays into float32 tensors.
    """
    
    def __init__(self, features: pd.DataFrame | np.ndarray, targets: pd.DataFrame | np.ndarray):
        """
        Initializes the dataset.
        
        Args:
            features: A DataFrame or numpy array containing the input features.
            targets: A DataFrame or numpy array containing the One-Hot encoded target variables.
            
        Raises:
            ValueError: If the number of samples in features and targets do not match.
        """
        if len(features) != len(targets):
            raise ValueError("The number of samples in features and targets must be identical.")
            
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.DataFrame):
            targets = targets.values
            
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            The dataset size.
        """
        return len(self.x)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.
        
        Args:
            idx: The index of the sample to retrieve.
            
        Returns:
            A tuple containing the feature tensor and the target tensor for the given index.
        """
        return self.x[idx], self.y[idx]
