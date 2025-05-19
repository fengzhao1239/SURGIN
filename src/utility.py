import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np



def load_hdf5(file_path):
    print(f">>>>> Loading data from {file_path}")
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            val = f[key][:-500]
            data.update({key: val})
            print(f">>>>> Loaded {key} with shape {val.shape}")
    return data

class OperatorDataset(Dataset):
    def __init__(self, data):
        perm = data['permeability_log']
        ts = data['time_step']
        sat = data['saturation']
        pre = data['pressure']
        
        normed_perm = self._min_max_normalize(perm, 'perm_min_max.json')
        normed_ts = self._min_max_normalize(ts, 'ts_min_max.json')
        normed_sat = self._min_max_normalize(sat, 'sat_min_max.json')
        normed_pre = self._min_max_normalize(pre, 'pre_min_max.json')
        
        num_ts = normed_ts.shape[-1]
        num_x = normed_sat.shape[-2]
        num_y = normed_sat.shape[-1]
        perm_tiled = np.tile(normed_perm[:, None, :, :], (1, num_ts, 1, 1))
        ts_tiled = np.tile(normed_ts[:, :, None, None], (1, 1, num_x, num_y))
        assert perm_tiled.shape == ts_tiled.shape == normed_sat.shape == normed_pre.shape,\
            f"Shapes mismatch: {perm_tiled.shape}, {ts_tiled.shape}, {normed_sat.shape}, {normed_pre.shape}"
        
        x = np.stack((perm_tiled, ts_tiled), axis=-1)
        y = np.stack((normed_sat, normed_pre), axis=-1)
        
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.x = self.x.permute(0, 2, 3, 1, 4)
        self.y = self.y.permute(0, 2, 3, 1, 4)
        self.num_samples = self.x.shape[0]
        print(f">>>>> Loaded dataset, x shape: {self.x.shape}, y shape: {self.y.shape}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def _min_max_normalize(self, data, save_name):
        min_val = np.min(data)
        max_val = np.max(data)
        save_min_max = {
            'min': float(min_val),
            'max': float(max_val)
        }
        with open(f'dataset/{save_name}', 'w') as f:
            json.dump(save_min_max, f, indent=4)
        return 2 * (data - min_val) / (max_val - min_val) - 1
        