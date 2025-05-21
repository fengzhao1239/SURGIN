import torch
import numpy as np
from einops import rearrange
import h5py


class Multi_Cartesian_Dataset:
    def __init__(self, hdf5_path, train_val_split=0.9):
        
        with h5py.File(hdf5_path, 'r') as f:
            permeability = f['permeability_log'][:-500]
        
        
        self.permeability = torch.from_numpy(permeability).float()
        self.permeability = self.permeability.unsqueeze(1)
        self.max = torch.max(self.permeability)
        self.min = torch.min(self.permeability)
        print(f"-- Whole dataset shape: {self.permeability.shape}, Max: {self.max}, Min: {self.min}")
        self.train_val_split = train_val_split
    
    def create_dataset(self):
        """
        Create a dataset by splitting the latent images into training and validation sets.
        """
        whole_dataset = self.permeability
        train_data, val_data = self._split_dataset(whole_dataset)
        print(f"-- Training data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
        
        norm_train_data = self._normalize(train_data, self.min, self.max)
        norm_val_data = self._normalize(val_data, self.min, self.max)
        
        return norm_train_data, norm_val_data
    
    def _split_dataset(self, whole_dataset):
        """
        Split the dataset into training and validation sets.
        """
        num_samples = whole_dataset.shape[0]
        split_index = int(num_samples * self.train_val_split)
        
        train_data = whole_dataset[:split_index]
        val_data = whole_dataset[split_index:]
        
        return train_data, val_data
    
    def _normalize(self, data, minimum, maximum):
        return -1 + (data - minimum) * 2. / (maximum - minimum)

